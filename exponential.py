import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

file_path = ('./Anvith/TC9/June1.csv')
data = pd.read_csv(file_path)

def grid_search_ses(series, alphas):
    best_alpha = None
    best_error = float("inf")

    for alpha in alphas:

        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)

        forecast = model.forecast(len(test))

        error = mean_squared_error(test, forecast)

        if error < best_error:
            best_error = error
            best_alpha = alpha

    return best_alpha, best_error


# Define the range of alpha values to search
alphas = np.linspace(0.01, 1.0, 100)

# Find the best alpha for each series
best_alpha_rainfall, best_error_rainfall = grid_search_ses(data['Rainfall'], alphas)
best_alpha_tmax, best_error_tmax = grid_search_ses(data['Tmax'], alphas)
best_alpha_tmin, best_error_tmin = grid_search_ses(data['Tmin'], alphas)

print(f"Best alpha for Rainfall: {best_alpha_rainfall:.2f} with MSE: {best_error_rainfall:.2f}")
print(f"Best alpha for Tmax: {best_alpha_tmax:.2f} with MSE: {best_error_tmax:.2f}")
print(f"Best alpha for Tmin: {best_alpha_tmin:.2f} with MSE: {best_error_tmin:.2f}")


# Function to apply Simple Exponential Smoothing with the best alpha
def apply_ses(series, alpha):
    model = SimpleExpSmoothing(series)
    model_fit = model.fit(smoothing_level=alpha, optimized=False)
    return model_fit

rainfall_model = apply_ses(data['Rainfall'], best_alpha_rainfall)
tmax_model = apply_ses(data['Tmax'], best_alpha_tmax)
tmin_model = apply_ses(data['Tmin'], best_alpha_tmin)


rainfall_forecast = rainfall_model.forecast(1).iloc[0]
tmax_forecast = tmax_model.forecast(1).iloc[0]
tmin_forecast = tmin_model.forecast(1).iloc[0]
r_data = pd.DataFrame({'rainfall': [rainfall_forecast]})

print(f"Predicted Rainfall for June 2024: {rainfall_forecast:.2f} mm")
print(f"Predicted Tmax for June 2024: {tmax_forecast:.2f} °C")
print(f"Predicted Tmin for June 2024: {tmin_forecast:.2f} °C")
