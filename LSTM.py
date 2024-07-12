import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

data = pd.read_csv('Harsha Vardhan/Harsha Vardhan/TC4/June1.csv')

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Rainfall', 'Tmax', 'Tmin']])

# Create lag features and prepare data for LSTM
def create_lagged_features(data, lag=30):
    X, y = [], []
    for i in range(lag, len(data)):
        X.append(data[i-lag:i])
        y.append(data[i])
    return np.array(X), np.array(y)

lag = 30
X, y = create_lagged_features(scaled_data, lag)


train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(3))  # Predicting 3 features: Rainfall, Tmax, Tmin
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=200, validation_data=(X_test, y_test), verbose=1)

# Make predictions
predictions = model.predict(X_test)

# Inverse transform predictions
predictions = scaler.inverse_transform(predictions)

# Display predictions
print(predictions)

# Prepare input for future prediction
future_input = scaled_data[-lag:]  # Last 'lag' days of data

# Predict for the next 30 days (or any desired period)
future_predictions = []
for _ in range(30):  # Predicting for one year
    future_input = future_input.reshape((1, lag, scaled_data.shape[1]))
    future_pred = model.predict(future_input)
    future_predictions.append(future_pred[0])
    future_input = np.append(future_input[:, 1:, :], future_pred.reshape(1, 1, 3), axis=1)

# Inverse transform future predictions
future_predictions = scaler.inverse_transform(future_predictions)

# Convert future predictions to DataFrame
future_predictions_df = pd.DataFrame(future_predictions, columns=['Rainfall', 'Tmax', 'Tmin'])

# Display future predictions
print(future_predictions_df)
