import pandas as pd
import numpy as np
import pickle

def load_city_data():
    return pd.read_csv('/Users/alexandreribeiro/Documents/GitHub/final_project/data/clean/data_with_cities.csv')

def load_models():
    with open('stacked_model.pkl', 'rb') as f:
        models = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return models['base_models'], models['meta_model'], scaler

def preprocess_input(city_metrics, user_inputs, scaler):
    input_data = {
        'population_city': city_metrics['population_city'],
        'greenspacearea_km2': city_metrics['greenspacearea_km2'],
        'AQI': city_metrics['AQI'],
        'adjusted_obesity_rate': 0.3 + (0.1 if user_inputs['obesity'] == "Yes" else 0),
        'adjusted_smoking_rate': 0.2 + (0.1 if user_inputs['smoking'] == "Yes" else 0),
        'adjusted_copd_rate': 0.05 + (0.05 if user_inputs['copd'] == "Yes" else 0),
        'adjusted_depression_rate': 0.15 + (0.1 if user_inputs['depression'] == "Yes" else 0)
    }
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    return scaler.transform(input_array)

def make_prediction(preprocessed_input, models):
    base_models, meta_model = models
    base_predictions = np.column_stack([model.predict(preprocessed_input) for model in base_models])
    final_prediction = meta_model.predict(base_predictions)
    return final_prediction[0]