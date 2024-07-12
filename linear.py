import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
df = pd.read_csv('June1.csv')

# Features and target
X = df[['Year']]
y_rainfall = df['Rainfall(mm)']
X_train, X_test, y_train_rainfall, y_test_rainfall = train_test_split(X, y_rainfall, test_size=0.2, random_state=42)


linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train_rainfall)


rainfall_predictions = linear_regression_model.predict(X_test)


rmse = root_mean_squared_error(y_test_rainfall, rainfall_predictions)
print(f'Linear Regression - RMSE: {rmse}')


linear_regression_model.fit(X, y_rainfall)
X_2024 = pd.DataFrame({'Year': [2024]})
predicted_rainfall_2024 = linear_regression_model.predict(X_2024)
print(f'Predicted Rainfall for 2024: {predicted_rainfall_2024[0]} mm')
