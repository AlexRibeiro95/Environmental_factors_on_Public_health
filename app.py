import streamlit as st
from backend import prepare_input, make_prediction, original_data
import requests
import os
import openai
from dotenv import load_dotenv
import seaborn as sns
import folium
from streamlit_folium import folium_static
import pandas as pd
import plotly.express as px
from openai import OpenAIError

# Load environment variables from .env file

load_dotenv()

# Load your API key securely

api_key = os.getenv("GPT_API_KEY")
openai.api_key = api_key

def get_suggestions_from_openai(smoker, copd, obesity, depression, max_tokens=200):
    prompt = f"Based on the user's health data, generate health suggestions: smoker: {smoker}, copd: {copd}, obesity: {obesity}, depression: {depression}."
    
    try:
        # Initial request with the updated API method
        response = openai.Completion.create(
            engine="text-davinci-003",  # Use "gpt-3.5-turbo" if you have access to ChatGPT models
            prompt=prompt,
            temperature=0.7,
            max_tokens=max_tokens  # Provide extra tokens as a buffer
        )
        
        # Parse the response
        suggestions = response.choices[0].text.strip()
        
        # Check if the suggestions end with a complete sentence
        if not suggestions.endswith('.'):
            # Follow-up request to complete the sentence
            follow_up_prompt = "Please complete the previous suggestions."
            follow_up_response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=follow_up_prompt,
                temperature=0.7,
                max_tokens=100  # Smaller token count for the follow-up
            )
            follow_up_suggestions = follow_up_response.choices[0].text.strip()
            suggestions += " " + follow_up_suggestions
        
        return suggestions
    
    except OpenAIError as e:
        # Handle any errors from the API
        st.error(f"Error fetching suggestions: {e}")
        return "Failed to retrieve suggestions. Please try again later."
    
    except Exception as e:
        # Handle any unexpected errors
        st.error(f"Unexpected error: {e}")
        return "An unexpected error occurred. Please try again later."

# Set the page layout to wide
st.set_page_config(layout="centered")

# Apply custom CSS for background color
st.markdown(
    """
    <style>
    /* Set the background color for the entire page */
    .stApp {
        background-color: #3C4F5B; /* Light grey color, replace with your desired color */
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

# Custom CSS to adjust the sidebar content
st.markdown(
    """
    <style>
    /* Adjust the position and styling of the sidebar radio buttons */
    .stRadio > div {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stRadio > div > label {
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 5px;
    }
    .stRadio > div > div {
        display: flex;
        justify-content: center;
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# Display the logo in the sidebar
st.sidebar.image("visualizations/logo.png", use_column_width=True)

# Add a centered and italic legend or caption below the logo
st.sidebar.markdown("""
<div style="text-align: center;">
    <em>We cannot control the future,</em><br>
    <em>but we can control the actions</em><br>
    <em>that will dictate the future.</em>
</div>
""", unsafe_allow_html=True)

# Continue with the rest of your sidebar content
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: -50px;">
    <h3>Navigation</h3>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio("", ["Project Overview", "Data Exploration", "Findings","Machine Learning", "Calculator"])

# Page 1: Project Overview
if page == "Project Overview":
    st.image("visualizations/head_image.webp", use_column_width=True)
    st.title("Project Overview")

    st.write("""
    ### How do we got here?

    In recent years, we have witnessed an alarming rise in various health conditions worldwide. Obesity, depression, respiratory illnesses, and chronic diseases are increasingly prevalent, affecting millions of lives and straining healthcare systems globally. These issues are further exacerbated by environmental challenges, such as air pollution and climate change, which are becoming more severe each year.

    As we grapple with these mounting health crises, urban green spaces—parks, forests, and gardens—emerge as a vital, yet often overlooked, solution. Green spaces have been shown to provide numerous health benefits, from improving mental well-being to reducing the risk of chronic diseases. They also play a crucial role in mitigating the effects of climate change, enhancing air quality, and fostering biodiversity within urban areas.
    """)

    st.write("""
    ### The Potential of Green Spaces

    The benefits of green spaces extend beyond individual health. By increasing the presence of greenery in our cities, we can contribute to the fight against climate change, reduce urban heat islands, and support a diverse range of plant and animal species. Green spaces also encourage physical activity, provide opportunities for social interaction, and improve overall quality of life in urban environments.

    Recognizing these multifaceted benefits, we embarked on this project to explore the potential impact of green spaces on public health. Our goal was to quantify the relationship between green spaces and various health metrics, such as life expectancy, obesity rates, and respiratory conditions, across different cities.
    """)

    st.write("""
    ### Our Journey: Developing the Project

    With these objectives in mind, we began by gathering extensive data from various public health and environmental sources. This data included metrics on air quality, green space coverage, population demographics, and health conditions across multiple cities.

    We knew that a robust data analysis framework was necessary to uncover meaningful insights from this wealth of information. To this end, we employed advanced machine learning techniques to analyze the data, identify patterns, and predict health outcomes based on the presence and quality of green spaces.

    Through this project, we aimed to provide actionable insights for urban planners, policymakers, and public health officials. By highlighting the critical role of green spaces in improving public health and combating environmental challenges, we hope to inspire more informed decisions that will lead to healthier, more sustainable cities.
    """)

    st.write("""
    ### What Lies Ahead

    In the following sections, we will delve into the findings from our analysis, discussing the impact of green spaces on public health and the environment. We will also walk you through the machine learning process we used to generate our predictions and provide a detailed look at the results of our models.

    Join us as we explore the data, uncover the connections between green spaces and health outcomes, and envision a future where our cities are healthier, greener, and more resilient.
    """)

# Page 2: Data Exploration
if page == "Data Exploration":
    st.title("Data Exploration")

    # Introduction to the page
    st.write("""
    ### Explore Public Health Metrics Across Cities

    This section allows you to explore various public health metrics collected from different cities. 
    You can select specific metrics, compare them across different cities, and even visualize the data on an interactive map.
    The adjusted metrics refer to a city level adjustment based on the national average.

    Use the options below to start exploring the data. You can select a metric to visualize across all cities, compare specific cities, or see how these metrics are distributed geographically.
    """)

    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/AlexRibeiro95/Environmental_factors_on_Public_health/main/data/clean/final_dataset.csv')

    # Display the dataset
    st.write("#### Full Dataset")
    st.dataframe(df)

    # Add a download button for the dataset
    st.download_button(
        label="Download Full Dataset",
        data=df.to_csv().encode('utf-8'),
        file_name='final_dataset.csv',
        mime='text/csv',
    )

    # List of health metrics available in the dataset
    metrics = [
        'adjusted_obesity_rate',
        'adjusted_smoking_rate', 'adjusted_exercising_rate', 'adjusted_chronic_rate',
        'adjusted_life_expectancy', 'adjusted_copd_rate', 'adjusted_depression_rate',
        'life_expectancy', 'AQI'
    ]

    # User selects a metric to explore
    selected_metric = st.selectbox("Select a health metric to visualize:", metrics)

    # Show the metric on a map
    map_center = [df['lat'].mean(), df['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=5)

    for _, row in df.iterrows():
        folium.CircleMarker(
            location=[row['lat'], row['lng']],
            radius=row[selected_metric] / 10,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=0.6,
            popup=f"{row['city']}: {row[selected_metric]}"
        ).add_to(m)

    folium_static(m)

    # Multi-select for cities to compare
    selected_cities = st.multiselect("Select cities to compare:", df['city'].unique())

    if selected_cities:
        # Filter data for the selected cities
        filtered_data = df[df['city'].isin(selected_cities)]
        
        # Display the comparison using Plotly Express
        fig = px.bar(filtered_data, x='city', y=selected_metric, title=f"Comparison of {selected_metric} for Selected Cities")
        fig.update_layout(xaxis_title="City", yaxis_title=selected_metric.replace('_', ' ').title())
        st.plotly_chart(fig)

# Page 3: Findings
elif page == "Findings":
    st.title("Findings")

    st.write("""
    ### The Multiple Benefits of Green Spaces

    Urban green spaces, such as parks and tree-lined streets, are widely recognized for their numerous benefits to both the environment and public health. As illustrated below, research has shown that the presence of urban trees and greenery can lead to:

    - **Reducing Rates of Cardiac Disease, Strokes, and Asthma**: Improved air quality can decrease the prevalence of respiratory and cardiovascular conditions.
    - **Protecting Biodiversity**: Urban green spaces provide essential habitats for birds, pollinators, and other wildlife.
    - **Reducing Obesity Levels**: Access to green spaces encourages physical activity, including walking and cycling.
    - **Managing Stormwater**: Green spaces help to control runoff, reduce flooding, and keep pollutants out of waterways.
    - **Cooling Urban Areas**: Trees can lower city temperatures, reducing heat-related illnesses and deaths.
    - **Filtering Air Pollutants**: Vegetation can trap fine particles, improving air quality within urban environments.
    - **Increasing Property Values**: Neighborhoods with abundant green spaces tend to have higher property values.
    - **Reducing Stress and Enhancing Mental Health**: Green spaces provide a natural environment that can alleviate stress, anxiety, and depression.
    """)

    st.image("visualizations/greenspaces_benefits.png", caption="Green Spaces Benefits", use_column_width=True)

    st.write("""
    ### Our Key Findings

    While our project aimed to uncover the impact of green spaces on life expectancy, the results were more nuanced than anticipated. Although we couldn't establish a direct correlation between green space coverage and life expectancy, our analysis provided valuable insights into the complex interplay between environmental factors and health outcomes.
    """)

    st.write("""
    #### 1. Air Quality Index (AQI) and Life Expectancy

    One of the most significant findings from our study was the strong relationship between the Air Quality Index (AQI) and life expectancy. Cities with higher AQI, indicating poorer air quality, were consistently associated with lower life expectancies. This finding underscores the importance of air quality in urban planning and public health initiatives.
    """)

    st.write("""
    #### 2. Health Conditions and Life Expectancy

    In addition to AQI, our analysis identified several health conditions that showed a strong correlation with life expectancy. Specifically, obesity rates, smoking rates, and the prevalence of chronic diseases were key factors influencing life expectancy across different cities.

    - **Obesity Rates**: Higher obesity rates were linked to lower life expectancy, reflecting the well-known health risks associated with obesity.
    - **Smoking Rates**: Smoking remains one of the most significant predictors of reduced life expectancy, highlighting the need for ongoing public health campaigns to reduce smoking prevalence.
    - **Chronic Diseases**: The prevalence of chronic diseases such as COPD and depression also showed a notable impact on life expectancy, emphasizing the need for comprehensive healthcare strategies to manage these conditions.
    """)

    st.write("""
    ### Conclusion

    While the direct impact of green spaces on life expectancy remains elusive, our findings highlight the importance of improving air quality and addressing key health conditions to enhance public health outcomes. The insights gained from this analysis can inform future research and urban planning efforts, helping to create healthier, more sustainable cities.
    """)

# Page 4: Machine Learning
elif page == "Machine Learning":
    st.title("Machine Learning")
    st.image("visualizations/ML_pipeline.png", caption="The Journey of Model Building", use_column_width=True)
    st.write("""
    ### Introduction

    The journey from raw data to a predictive model that can estimate life expectancy based on various health metrics and environmental factors involves several critical steps. Below is a summary of the entire machine learning pipeline that we developed for this project.
    """)

    st.write("""
    ### 1. Data Acquisition

    Our first step was gathering a comprehensive dataset that includes various health and environmental metrics across multiple cities. The data acquisition process involved:

    - **City Data Retrieval**: Collecting basic information about each city, including coordinates and population size.
    - **Google Earth Engine (GEE)**: Extracting data on green space coverage in each city, which is critical to understanding the potential environmental impact on health.
    - **Health Data Collection**: Gathering key health metrics such as obesity rates, smoking rates, exercise levels, life expectancy, and chronic conditions (including COPD and depression) from reliable public health databases.
    - **OpenWeather API**: Collecting Air Quality Index (AQI) data to include as a key environmental variable that could affect health outcomes.
    """)

    st.write("""
    ### 2. Data Cleaning and Exploratory Data Analysis (EDA)

    With the raw data in hand, the next step was to clean and prepare it for analysis. This involved:

    - **Handling Missing Values**: We identified and filled or removed missing values to ensure the integrity of our dataset.
    - **Data Type Conversion**: Ensuring that all data was in the correct format, particularly when dealing with numerical and categorical variables.
    - **Outlier Detection and Handling**: Identifying and managing outliers to prevent them from skewing our results. This included techniques like the Interquartile Range (IQR) and Z-score methods.
    - **Correlation Analysis**: Using correlation matrices to identify relationships between variables, which helped guide feature selection and engineering.
    """)

    st.image("visualizations/Correlation_matrix_features.png", caption="Correlation Matrix", use_column_width=True)

    st.write("""
    ### 3. Feature Engineering

    To enhance the predictive power of our models, we performed several feature engineering tasks:

    - **Creating New Features**: We developed new features based on existing health metrics, such as adjusted rates for various conditions at the city level. We also incorporated environmental features like AQI and green space coverage.
    - **Feature Selection**: Based on correlation coefficients, we selected the most relevant features while avoiding highly correlated ones to prevent multicollinearity.
    - **Data Transformation**: Standardizing and scaling the data to ensure that all features contributed equally to the model and that no single feature dominated due to its scale.
    """)

    st.write("""
    ### 4. Model Selection and Tuning

    The heart of the machine learning process involved selecting the best models, tuning them, and validating their performance:

    - **Model Selection**: We experimented with several regression models, including:
      - Linear Regression
      - Support Vector Regression (SVR)
      - Gradient Boosting
      - Random Forest Regressor
      - Decision Tree Regressor
      - XGBoost

    - **Cross-Validation**: We used cross-validation to ensure that our models were not overfitting and could generalize well to unseen data.
    - **GridSearchCV**: For the top-performing models, we applied GridSearchCV to find the best hyperparameters, further optimizing the models.
    - **Overfitting Check**: We trained the models on the training set and evaluated them on the test set to check for overfitting. If the performance on the test set was significantly worse, we made adjustments to mitigate overfitting.
    """)

    st.write("""
    ### 5. Stacking and Final Model

    To maximize predictive accuracy, we employed a stacking technique:

    - **Model Stacking**: We combined the predictions of the best-performing models into a final ensemble model. This model benefits from the strengths of each individual model.
    - **Final Model Training**: We trained the final stacked model on the full dataset, ensuring it had the best possible chance of making accurate predictions.
    - **SHAP Analysis**: Finally, we applied SHAP (SHapley Additive exPlanations) to interpret the model's predictions. SHAP values allowed us to understand the contribution of each feature to the model’s predictions, providing transparency and insight into the factors most influencing life expectancy.
    """)

    st.image("visualizations/SHAP_features.png", caption="SHAP Feature Importance", use_column_width=True)

    st.write("""
    ### Conclusion

    Through this rigorous and comprehensive machine learning process, we developed a robust model capable of predicting life expectancy based on a wide range of health and environmental factors. The model's performance metrics and SHAP analysis provide valuable insights into the key drivers of life expectancy, offering potential pathways for improving public health outcomes.
    """)

    st.image("visualizations/Grouplot_final_staked_model.png", caption="Final Performance Results", use_column_width=True)

# Page 5: Calculator
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
    
    