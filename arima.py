import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

df=pd.read_csv('./Anvith/01-05_to_17-06.csv')
df.set_index(['Longitude', 'Latitude'], inplace=True)
d = list(df.iloc[0, 0:33])
column_names=['Rainfall']
data=pd.DataFrame(d,columns=column_names)
train=data[:-12]
test=data[-12:]
# plot_pacf(data)
# plt.show()
model=ARIMA(train['Rainfall'],order=(7,0,1))
result=model.fit()
start= len(train)
end = len(train)+len(test)-1
pred = result.predict(start =start,end = end)
print(pred)
plt.figure(figsize=(10, 6))
plt.plot(train.index, train['Rainfall'], label='Train Data')
plt.plot(test.index, test['Rainfall'], label='Test Data')
plt.plot(test.index, pred, label='Predicted Data', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Rainfall')
plt.title('ARIMA Model - Predicted vs Actual Rainfall')
plt.legend()
plt.show()
rmse=root_mean_squared_error(test,pred)
print(rmse)


