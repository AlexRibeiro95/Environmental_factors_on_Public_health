import streamlit as st
from backend import prepare_input, make_prediction, original_data
import requests
import os
import openai
from dotenv import load_dotenv
import seaborn as sns
import folium
from streamlit_folium import folium_static

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
st.sidebar.image("/Users/alexandreribeiro/Documents/GitHub/final_project/visualizations/logo.png", use_column_width=True)

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

page = st.sidebar.radio("", ["Project Overview", "Data Exploring","Findings","Machine Learning", "Calculator"])

# Page 1: Project Overview
if page == "Project Overview":
    st.title("Project Overview")
    st.write("""
    ### Why This Project?

    In recent years, the intersection of urban planning and public health has garnered significant attention. Green spaces, such as parks, gardens, and natural landscapes, are increasingly recognized for their potential to improve physical and mental well-being. However, despite widespread belief in their benefits, the direct impact of green spaces on public health remains underexplored, particularly when it comes to measurable health outcomes.

    This project was conceived to investigate the relationship between green spaces and public health metrics. Initially, we aimed to prove a direct correlation between the availability of green spaces in urban areas and improved health outcomes. However, as the project progressed, it became clear that other environmental factors, particularly the Air Quality Index (AQI), played a significant role in this complex relationship.

    ### How AQI Came into Focus

    As we delved deeper into the data, it became evident that air quality might mediate the relationship between green spaces and health. Poor air quality is a well-documented risk factor for a variety of health issues, and it can diminish the potential benefits of green spaces. Therefore, AQI became a critical variable in our analysis, allowing us to explore not only the direct impact of green spaces but also how air quality might influence this relationship.
    """)

# Page 2: Data Exploring
if page == "Data Exploration":
    st.title("Data Exploration")

    # Introduction to the page
    st.write("""
    ### Explore Public Health Metrics Across Cities

    This section allows you to explore various public health metrics collected from different cities. 
    You can select specific metrics, compare them across different cities, and even visualize the data on an interactive map.

    Use the options below to start exploring the data. You can select a metric to visualize across all cities, compare specific cities, or see how these metrics are distributed geographically.
    """)


    # Load the dataset
    df = pd.read_csv('https://raw.githubusercontent.com/AlexRibeiro95/Environmental_factors_on_Public_health/main/data/clean/final_dataset.csv')

    # List of health metrics available in the dataset
    metrics = [
        'obesity_rate', 'smoking_rate', 'exercising_rate', 'adjusted_obesity_rate',
        'adjusted_smoking_rate', 'adjusted_exercising_rate', 'adjusted_chronic_rate',
        'adjusted_life_expectancy', 'adjusted_copd_rate', 'adjusted_depression_rate',
        'life_expectancy', 'AQI'
    ]

    # User selects a metric to explore
    selected_metric = st.selectbox("Select a health metric to visualize:", metrics)

    # Plot the selected metric for all cities
    plt.figure(figsize=(10, 6))
    sns.barplot(x='city', y=selected_metric, data=df)
    plt.xticks(rotation=90)
    plt.title(f"Comparison of {selected_metric} Across Cities")
    plt.ylabel(selected_metric.replace('_', ' ').title())
    plt.xlabel("City")
    st.pyplot(plt)

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
        
        # Display the comparison
        st.write(filtered_data[['city'] + metrics])
        
        # Plot the selected metric for these cities
        plt.figure(figsize=(10, 6))
        sns.barplot(x='city', y=selected_metric, data=filtered_data)
        plt.title(f"Comparison of {selected_metric} for Selected Cities")
        plt.ylabel(selected_metric.replace('_', ' ').title())
        plt.xlabel("City")
        st.pyplot(plt)

# Page 3: Findings
elif page == "Findings":
    st.title("Findings")
    st.image("visualizations/greenspaces_benefits.png", caption="Green Spaces Benefits", use_column_width=True)
    st.write("""
    ### Key Findings

    Throughout our analysis, we faced the challenge of not being able to definitively prove that green spaces have a direct impact on public health. This was a surprising result, given the common assumption that more green space equates to better health. However, this finding doesn't diminish the importance of green spaces; rather, it highlights the complexity of their impact on health.

    **Why Couldn’t We Prove the Direct Impact?**

    The lack of a clear, direct relationship could be due to several factors:
    - **Data Limitations**: The available data may not have captured all relevant aspects of how green spaces are used or their quality.
    - **Confounding Variables**: Other factors, such as socioeconomic status, urban density, and particularly air quality, may have stronger or more direct impacts on health outcomes.
    - **Complex Interactions**: The benefits of green spaces might be mediated or moderated by other factors, making the relationship difficult to isolate.

    ### Proven Benefits of Green Spaces

    Despite the challenges in proving a direct impact, extensive research supports the numerous benefits of green spaces in urban areas, including:
    - **Physical Health**: Regular access to green spaces encourages physical activity, which is linked to reduced risks of obesity, cardiovascular diseases, and other chronic conditions.
    - **Mental Health**: Green spaces have been shown to reduce stress, anxiety, and depression, providing a natural environment for relaxation and mental rejuvenation.
    - **Social Benefits**: Green spaces offer a communal area for social interaction, fostering community ties and improving overall quality of life.

    These findings suggest that while the direct impact on specific health metrics may be complex and influenced by multiple factors, the overall contribution of green spaces to public well-being is significant and cannot be overlooked.
    """)

# Page 4: Machine Learning
elif page == "Machine Learning":
    st.title("Machine Learning")
    st.image("visualizations/ML_pipeline.png", caption="The Journey of Model Building", use_column_width=True)
    st.write("""
    ### Our Approach

    To explore the relationship between green spaces, AQI, and public health, we employed a structured machine learning approach. This section provides an overview of the key steps we took:

    1. **Data Collection and Preprocessing**:
    - We gathered data from multiple sources, including public health records, environmental datasets, and urban planning databases.
    - Data preprocessing involved cleaning the data, handling missing values, and engineering features that might capture the complex interactions between green spaces and health outcomes.

    2. **Model Selection and Training**:
    - We explored several machine learning models, including Linear Regression, Decision Trees, Random Forests, and XGBoost, to identify the best approach for our data.
    - The models were trained using a combination of historical data and relevant environmental variables.

    3. **Hyperparameter Tuning and Regularization**:
    - To optimize the performance of our models, we conducted extensive hyperparameter tuning. This process involved adjusting model parameters to minimize errors and improve predictive accuracy.
    - Regularization techniques were applied to prevent overfitting, ensuring that our models could generalize well to unseen data.

    4. **Evaluation and Interpretation**:
    - The models were evaluated using key metrics such as R² Score, Mean Absolute Error (MAE), and Mean Squared Error (MSE).
    - Visualization of feature importance and error analysis helped us interpret the results and refine our understanding of the underlying relationships.

    ### Results

    The final stacked model, which combined the strengths of multiple models, demonstrated strong predictive capabilities, as shown in the graphs below:
    """)

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
    
    