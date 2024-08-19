import streamlit as st
import pickle
import numpy as np

# Function to load the stacked model
def load_stacked_model():
    with open('stacked_model.pkl', 'rb') as file:
        models = pickle.load(file)
    return models['base_models'], models['meta_model']

# Main function for the Streamlit app
def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "Process Overview", "Findings", "Life Expectancy Calculator"])

    if page == "Introduction":
        st.title("Introduction")
        st.write("""
            Welcome to the Life Expectancy Predictor. This application allows you to estimate life expectancy based on several public health metrics and environmental factors.
            """)
    
    elif page == "Process Overview":
        st.title("Process Overview")
        st.write("""
            This section provides an overview of the data processing and modeling techniques used to create the life expectancy predictor.
            """)
    
    elif page == "Findings":
        st.title("Findings")
        st.write("""
            Here we summarize the key findings and insights derived from our analysis.
            """)
    
    elif page == "Life Expectancy Calculator":
        st.title("Life Expectancy Calculator")
        
        # Load the stacked model
        base_models, meta_model = load_stacked_model()
        
        st.header("Input Parameters")
        
        # City and Environment Section
        st.subheader("City and Environment")
        city_population = st.selectbox(
            "City Population",
            ("Small (1,000 - 50,000)", "Medium (50,000 - 1,000,000)", "Large (1,000,000+)"),
            help="Select the population size of the city."
        )
        greenspace = st.selectbox(
            "Green Space Availability",
            ("Low (<20 km²)", "Moderate (20-100 km²)", "High (>100 km²)"),
            help="Select the category that best describes the green space area in your city."
        )
        AQI = st.selectbox(
            "Air Quality Index (AQI)",
            ("Excellent (1-50)", "Good (51-100)", "Moderate (101-150)", "Poor (151-200)", "Very Poor (201+)"),
            help="Select the air quality level in your city."
        )

        # Health Metrics Section
        st.subheader("Health Metrics")
        obesity_rate = st.selectbox(
            "Obesity (BMI > 30)", 
            ("Yes", "No"),
            help="Do you have a BMI greater than 30?"
        )
        smoking_rate = st.selectbox(
            "Smoking", 
            ("Yes", "No"),
            help="Do you currently smoke?"
        )
        copd_rate = st.selectbox(
            "COPD", 
            ("Yes", "No"),
            help="Do you have chronic obstructive pulmonary disease (COPD)?"
        )
        depression_rate = st.selectbox(
            "Depression", 
            ("Yes", "No"),
            help="Do you suffer from depression?"
        )

        # Convert categorical inputs to numerical values
        population_mapping = {
            "Small (1,000 - 50,000)": 1, 
            "Medium (50,000 - 1,000,000)": 2, 
            "Large (1,000,000+)": 3
        }
        greenspace_mapping = {
            "Low (<20 km²)": 1,
            "Moderate (20-100 km²)": 2,
            "High (>100 km²)": 3
        }
        AQI_mapping = {
            "Excellent (1-50)": 1,
            "Good (51-100)": 2,
            "Moderate (101-150)": 3,
            "Poor (151-200)": 4,
            "Very Poor (201+)": 5
        }

        city_population = population_mapping[city_population]
        greenspacearea_km2 = greenspace_mapping[greenspace]
        AQI = AQI_mapping[AQI]
        obesity_rate = 1 if obesity_rate == "Yes" else 0
        smoking_rate = 1 if smoking_rate == "Yes" else 0
        copd_rate = 1 if copd_rate == "Yes" else 0
        depression_rate = 1 if depression_rate == "Yes" else 0

        # Prepare the input data for prediction
        input_data = np.array([[
            city_population, 
            greenspacearea_km2, 
            AQI, 
            obesity_rate, 
            smoking_rate, 
            copd_rate, 
            depression_rate
        ]])

        # Predict using the stacked model
        base_predictions = np.zeros((input_data.shape[0], len(base_models)))

        for i, (name, model) in enumerate(base_models):
            base_predictions[:, i] = model.predict(input_data)
        
        final_prediction = meta_model.predict(base_predictions)

        # Display the prediction
        st.write(f"## Predicted Life Expectancy: {final_prediction[0]:.2f} years")

if __name__ == "__main__":
    main()