import pickle
import numpy as np
import pandas as pd
import os
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Get the directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the scaler and the stacked model
with open(os.path.join(BASE_DIR, 'scaler.pkl'), 'rb') as f:
    scaler = pickle.load(f)

with open(os.path.join(BASE_DIR, 'stacked_model.pkl'), 'rb') as f:
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

def make_prediction(input_data, smoker, copd, obesity, depression):
    """
    Make a prediction using the stacked model and adjust it based on health conditions.
    """
    # Generate predictions from base models
    base_predictions = np.column_stack([model.predict(input_data) for model in base_models])

    # Use the meta model to make the final prediction
    prediction = meta_model.predict(base_predictions)[0]

    # Adjust the prediction based on the user's health metrics
    if smoker == "YES" and copd == "YES":
        prediction -= 15  # Strong adjustment for combined smoking and COPD
    elif copd == "YES":
        prediction -= 10  # COPD alone
    elif smoker == "YES":
        prediction -= 7  # Smoking alone

    if obesity == "YES":
        prediction -= 7  # Obesity adjustment

    if depression == "YES":
        prediction -= 1.5  # Depression adjustment

    # Ensure the prediction doesn't drop below a reasonable threshold
    prediction = max(prediction, 0)

    return prediction