import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error,r2_score
import numpy as np

# Load the dataset
data = pd.read_csv("June1.csv")

# Define features (X) and target (y)
X = data[['Year', 'Tmax', 'Tmin']]
y = data['Rainfall(mm)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)
r2=r2_score(y_test,y_pred)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", rmse)
print(r2)
