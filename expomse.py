import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import root_mean_squared_error


def grid_search_ses(series, alphas):
    best_alpha = None
    best_error = float("inf")

    for alpha in alphas:

        train_size = int(len(series) * 0.8)
        train, test = series[:train_size], series[train_size:]

        model = SimpleExpSmoothing(train).fit(smoothing_level=alpha, optimized=False)

        forecast = model.forecast(len(test))

        error = root_mean_squared_error(test, forecast)

        if error < best_error:
            best_error = error
            best_alpha = alpha

    return best_alpha, best_error


# Define the range of alpha values to search
alphas = np.linspace(0.01, 1.0, 100)

# Generate file paths for June data
file_paths = ['./Anvith/TC9/June{}.csv'.format(i) for i in range(1, 31)]

# Iterate over file paths
for file_path in file_paths:
    print(f"Processing file: {file_path}")

    # Read the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Perform grid search for each series
    best_alpha_rainfall, best_error_rainfall = grid_search_ses(data['Rainfall'], alphas)
    best_alpha_tmax, best_error_tmax = grid_search_ses(data['Tmax'], alphas)
    best_alpha_tmin, best_error_tmin = grid_search_ses(data['Tmin'], alphas)

    # Print results
    print(f"Best alpha for Rainfall: {best_alpha_rainfall:.2f} with MSE: {best_error_rainfall:.2f}")
    print(f"Best alpha for Tmax: {best_alpha_tmax:.2f} with MSE: {best_error_tmax:.2f}")
    print(f"Best alpha for Tmin: {best_alpha_tmin:.2f} with MSE: {best_error_tmin:.2f}")
    print()
