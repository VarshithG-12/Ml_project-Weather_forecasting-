import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
file_path = ('./Anvith/TC7-RMSE.csv')
   
data = pd.read_csv(file_path)

# Ensure the 'Date' column is in datetime format
data['Date'] = pd.to_datetime(data['Date'],format='%d-%m-%Y')

# Set 'Date' as the index
data.set_index('Date', inplace=True)


# data['Rainfall_rmse'] = np.sqrt(data['Rainfall'])
# data['Tmax_rmse'] = np.sqrt(data['Tmax'])
# data['Tmin_rmse'] = np.sqrt(data['Tmin'])

# Check the data structure
print(data.head())
print(data.info())

# Plot the results
plt.figure(figsize=(14, 10))

# Plot Rainfall
plt.subplot(3, 1, 1)
plt.plot(data.index, data['Rainfall'], label='Rainfall', color='blue')
plt.title('Rainfall RMSE')
plt.ylabel('RMSE')
plt.legend()

# Plot Tmax
plt.subplot(3, 1, 2)
plt.plot(data.index, data['Tmax'], label='Tmax', color='green')
plt.title('Tmax RMSE')
plt.ylabel('RMSE')
plt.legend()

# Plot Tmin
plt.subplot(3, 1, 3)
plt.plot(data.index, data['Tmin'], label='Tmin', color='red')
plt.title('Tmin RMSE')
plt.xlabel('Date')
plt.ylabel('RMSE')
plt.legend()

# Adjust x-axis to show dates from June 1 to June 30
plt.xlim([pd.to_datetime('2024-06-01', format='%Y-%m-%d',), pd.to_datetime('2024-06-30', format='%Y-%m-%d')])

plt.tight_layout()
plt.show()
