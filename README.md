# Environmental Factors on Public Health 🌍

## Table of Contents
1. [Introduction](#introduction)
2. [Objective](#objective)
3. [Data Collection](#data-collection)
4. [Data Preprocessing](#data-preprocessing)  
5. [Model Selection and Tuning](#model-selection-and-tuning)  
6. [Model Performance](#model-performance)    
7. [Feature Importance](#feature-importance)
8. [Deployment](#deployment)  
9. [Challenges and Future Work](#challenges-and-future-work) 
10. [Conclusion](#conclusion)
11. [References](#references)


---

## Introduction
Welcome to the Life Expectancy Prediction Project. This project aims to predict life expectancy based on various health metrics such as smoking status, obesity, air quality, and more. The model leverages advanced machine learning techniques to provide insights that can be used to improve public health strategies.

## Objective
The primary objective of this project is to develop a robust machine learning model that accurately predicts life expectancy using a combination of health, environmental, and demographic data.

## Data Collection

### Source of Data
- **SimpleMaps**: Provided city data including coordinates and population ([SimpleMaps](https://simplemaps.com/data/world-cities)).
- **Google Earth Engine (GEE)**: Used to obtain green space areas within city limits.
- **Centers for Disease Control and Prevention (CDC)**: Collected health-related data such as obesity rates, smoking rates, exercise rates, and more.
- **OpenWeather API**: Used to collect the Air Quality Index (AQI).

### Data Acquisition Process
- **City Data**: Retrieved from SimpleMaps, focusing on cities in the US.
- **Green Spaces**: Data obtained via Google Earth Engine using city coordinates.
- **Health Metrics**: Retrieved from the CDC’s database, aggregated at the state level.
- **Air Quality Index (AQI)**: Collected via OpenWeather API for each city.

## Data Preprocessing

### Data Cleaning
- **Handling Missing Values**: Imputed missing values where necessary.
- **Outlier Detection and Removal**: Identified and treated outliers in the dataset.
- **Converting Data Types**: Ensured all data types were appropriate for analysis.

### Feature Engineering
- **Creating New Features**: Developed features like logical inverse health metrics and adjusted health metrics at the city level.
- **Feature Selection**: Selected features based on correlation analysis to avoid multicollinearity.
- **Data Transformation**: Applied standardization and scaling to prepare the data for modeling.

### Exploratory Data Analysis (EDA)
- **Correlation Matrix**: Analyzed correlations between different features to understand relationships and dependencies.

## Model Selection and Tuning

### Models Used
- **Linear Regression**
- **Decision Tree Regressor**
- **Random Forest Regressor**
- **Gradient Boosting Regressor**
- **Support Vector Regressor (SVR)**
- **XGBoost Regressor**

### Hyperparameter Tuning
- **GridSearchCV**: Used to find the best hyperparameters for Random Forest, XGBoost, and Decision Tree models.
- **Cross-Validation**: Performed 5-fold cross-validation to validate model performance.

### Overfitting Check
- **Overfitting Mitigation**: Checked for overfitting by comparing performance on training and test data.

### Stacked Model
- **Meta-Model**: A Linear Regression model was used to stack the predictions of the best-performing models (Random Forest, XGBoost, Decision Tree).
- **Final Model Training**: The final stacked model was trained on the entire dataset.

## Model Performance

### Performance Metrics
- **Evaluation Metrics**: The models were evaluated using Mean Absolute Error (MAE), Mean Squared Error (MSE), and R² score.
- **Comparison of Models**: Detailed comparison of model performance metrics.

### Final Stacked Model Results
- **Training vs. Test Performance**: Comparison of performance metrics for the final stacked model on both training and test data.

## Feature Importance

### SHAP Analysis
- **SHAP Values**: Used to explain the impact of each feature on the model's predictions.
- **Key Features**: Identified the most significant features affecting life expectancy predictions.

## Deployment

### Streamlit Application
- **Overview**: Developed an interactive web application using Streamlit where users can input their data to predict life expectancy.
- **User Instructions**: Detailed instructions on how to use the app, interpret the results, and receive health suggestions.

## Challenges and Future Work

### Challenges Faced
- **Data Limitations**: Discussed challenges such as missing data, computational constraints, and model complexity.
- **Overfitting**: Addressed the challenge of overfitting in model training.

### Future Enhancements
- **Improving Model Accuracy**: Suggestions for incorporating additional data sources and refining the model.
- **Wider Deployment**: Plans to deploy the model in different environments for broader use.

## Conclusion

### Summary
Summarized the key findings and achievements of the project.

### Impact
Reflected on the potential impact of the project on public health awareness and policy-making.

## References
- **SimpleMaps**: https://simplemaps.com/data/world-cities
- **Google Earth Engine**: https://earthengine.google.com/
- **CDC Data**: https://cdi.cdc.gov/
- **OpenWeather API**: https://openweathermap.org/api


### Additional Visualizations
Provided extra visualizations that support the findings in the main body.
