import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import root_mean_squared_error

# Load the data
df = pd.read_csv('June1.csv')

# Features and targets
X = df[['Year']]
y_rainfall = df['Rainfall(mm)']
y_tmax = df['Tmax']
y_tmin = df['Tmin']

# Split the data
X_train, X_test, y_train_rainfall, y_test_rainfall = train_test_split(X, y_rainfall, test_size=0.2, random_state=42)
_, _, y_train_tmax, y_test_tmax = train_test_split(X, y_tmax, test_size=0.2, random_state=42)
_, _, y_train_tmin, y_test_tmin = train_test_split(X, y_tmin, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'ElasticNet Regression': ElasticNet(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'Support Vector Regressor': SVR(),
    'k-Nearest Neighbors': KNeighborsRegressor(),
    'XGBoost': XGBRegressor(random_state=42)
}

# Function to train and evaluate models
def train_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        rmse = root_mean_squared_error(y_test,predictions)
        results[name] = rmse
        print(f'{name} - RMSE: {rmse}')
    return results

# Train and evaluate models for each target variable
print("Rainfall Predictions:")
rainfall_results = train_evaluate(models, X_train, X_test, y_train_rainfall, y_test_rainfall)

print("\nTmax Predictions:")
tmax_results = train_evaluate(models, X_train, X_test, y_train_tmax, y_test_tmax)

print("\nTmin Predictions:")
tmin_results = train_evaluate(models, X_train, X_test, y_train_tmin, y_test_tmin)

# Choose the best models based on MSE results
best_model_rainfall = SVR()
best_model_tmax = RandomForestRegressor(n_estimators=100, random_state=42)
best_model_tmin = KNeighborsRegressor()

# Train the best models on the entire dataset
best_model_rainfall.fit(X, y_rainfall)
best_model_tmax.fit(X, y_tmax)
best_model_tmin.fit(X, y_tmin)

# Predict for 2024
X_2024 = pd.DataFrame({'Year': [2024]})

prain_2024 = best_model_rainfall.predict(X_2024)
ptmax_2024 = best_model_tmax.predict(X_2024)
ptmin_2024 = best_model_tmin.predict(X_2024)
print("\n")
print(f'Predicted Rainfall for 2024: {prain_2024[0]} mm')
print(f'Predicted Tmax for 2024: {ptmax_2024[0]} °C')
print(f'Predicted Tmin for 2024: {ptmin_2024[0]} °C')
