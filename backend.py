import pickle
import numpy as np
import pandas as pd

# Load the scaler and the stacked model
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('stacked_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    base_models = model_data['base_models']
    meta_model = model_data['meta_model']

# Load the original dataset
original_data = pd.read_csv('/Users/alexandreribeiro/Documents/GitHub/final_project/data/clean/data_with_cities.csv')

def get_city_features(city_name):
    """
    Retrieve the features corresponding to the selected city.
    """
    city_data = original_data[original_data['city'] == city_name]
    if city_data.empty:
        raise IndexError("City not found in the dataset.")
    city_data = city_data.iloc[0]
    return city_data['greenspacearea_km2'], city_data['AQI'], city_data['population_city']

def prepare_input(city_name, obesity, smoker, copd, depression):
    """
    Retrieve the standardized and scaled features from the original dataset
    based on the selected city and user's health status, ensuring 11 features without the target.
    """
    # Get the city-specific row from the original dataset
    city_data = original_data[original_data['city'] == city_name]
    
    if city_data.empty:
        raise IndexError("City not found in the dataset.")
    
    # Extract the relevant features based on the selected features list
    population = city_data['population_city'].values[0]
    greenspace = city_data['greenspacearea_km2'].values[0]
    AQI = city_data['AQI'].values[0]
    
    # Adjust health metrics based on user input and retrieve logical inverses
    obesity_rate = city_data['adjusted_obesity_rate'].values[0] if obesity == "YES" else -city_data['adjusted_obesity_rate'].values[0]
    smoking_rate = city_data['adjusted_smoking_rate'].values[0] if smoker == "YES" else -city_data['adjusted_smoking_rate'].values[0]
    copd_rate = city_data['adjusted_copd_rate'].values[0] if copd == "YES" else -city_data['adjusted_copd_rate'].values[0]
    depression_rate = city_data['adjusted_depression_rate'].values[0] if depression == "YES" else -city_data['adjusted_depression_rate'].values[0]

    # Logical inverse values
    logical_inverse_obesity = -obesity_rate
    logical_inverse_smoking = -smoking_rate
    logical_inverse_copd = -copd_rate
    logical_inverse_depression = -depression_rate

    # Combine all features into a single input vector
    input_vector = [
        population, greenspace, AQI, 
        obesity_rate, smoking_rate, copd_rate, depression_rate,
        logical_inverse_obesity, logical_inverse_smoking, 
        logical_inverse_copd, logical_inverse_depression
    ]

    # Convert to a NumPy array and reshape for the model input
    input_vector = np.array(input_vector).reshape(1, -1)

    # Ensure that the input vector has exactly 11 features
    assert input_vector.shape[1] == 11, f"Expected 11 features, got {input_vector.shape[1]} features."

    return input_vector

def make_prediction(input_data):
    """
    Make a prediction using the stacked model.
    """
    # Generate predictions from base models
    base_predictions = np.column_stack([model.predict(input_data) for model in base_models])

    # Use the meta model to make the final prediction
    prediction = meta_model.predict(base_predictions)

    # Since the target is not scaled, return the prediction directly
    return prediction[0]

    # Inverse transform to get the prediction in the original scale
    original_scale_prediction = target_scaler.inverse_transform(scaled_prediction.reshape(-1, 1))

    return original_scale_prediction[0][0]