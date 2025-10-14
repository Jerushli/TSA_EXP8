# Ex.No: 08     MOVING AVERAGE MODEL AND EXPONENTIAL SMOOTHING
### Date: 14-10-2025
#### NAME: JERUSHLIN JOSE JB
#### REGISTER NUMBER: 212222240039

### AIM:
To implement Moving Average Model and Exponential smoothing Using Python.
### ALGORITHM:
1. Import necessary libraries
2. Read the electricity time series data from a CSV file,Display the shape and the first 20 rows of
the dataset
3. Set the figure size for plots
4. Suppress warnings
5. Plot the first 50 values of the 'Value' column
6. Perform rolling average transformation with a window size of 5
7. Display the first 10 values of the rolling mean
8. Perform rolling average transformation with a window size of 10
9. Create a new figure for plotting,Plot the original data and fitted value
10. Show the plot
11. Also perform exponential smoothing and plot the graph
### PROGRAM:
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load data
data = pd.read_csv('/content/NFLX.csv')

# ✅ Make sure you have a datetime column (replace 'date' with your actual column name)
# Example: if your column is 'timestamp' or 'datetime', change it accordingly
data['Date'] = pd.to_datetime(data['Date'])

# ✅ Set datetime column as index
data.set_index('Date', inplace=True)

# Select 'open' column
passengers_data = data[['Open']]
print("Shape of the dataset:", passengers_data.shape)
print("First 10 rows of the dataset:")
print(passengers_data.head(10))

# Plot original data
plt.figure(figsize=(12, 6))
plt.plot(passengers_data['Open'], label='Original Open Data')
plt.title('Original Data')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.grid()
plt.show()

# Rolling mean
rolling_mean_5 = passengers_data['Open'].rolling(window=5).mean()
rolling_mean_10 = passengers_data['Open'].rolling(window=10).mean()

plt.figure(figsize=(12, 6))
plt.plot(passengers_data['Open'], label='Original Data', color='blue')
plt.plot(rolling_mean_5, label='Moving Average (window=5)')
plt.plot(rolling_mean_10, label='Moving Average (window=10)')
plt.title('Moving Averages of Open')
plt.xlabel('Date')
plt.ylabel('Open')
plt.legend()
plt.grid()
plt.show()

data_monthly = data.resample('MS').sum()  # MS = Month Start
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly['Open'].values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)

print(scaled_data.head())


from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Assuming 'data' already has datetime index from previous step
data_monthly = data.resample('MS').sum()
data_monthly = data_monthly.asfreq('MS')  # Ensure monthly frequency

# Scaling and shifting to avoid non-positive values
scaler = MinMaxScaler()
scaled_data = pd.Series(
    scaler.fit_transform(data_monthly['Open'].values.reshape(-1, 1)).flatten(),
    index=data_monthly.index
)
scaled_data = scaled_data + 1  # avoid zeros

# Train-test split
x = int(len(scaled_data) * 0.8)
train_data = scaled_data[:x]
test_data = scaled_data[x:]

# Model 1: Train and test visualization
model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

# Plot train vs test vs forecast
ax = train_data.plot(figsize=(10,5))
test_data.plot(ax=ax)
test_predictions_add.plot(ax=ax)
ax.legend(["Train Data", "Test Data", "Forecast"])
ax.set_title('Holt-Winters Exponential Smoothing (Train/Test Split)')
plt.grid()
plt.show()

# RMSE Evaluation
rmse = np.sqrt(mean_squared_error(test_data, test_predictions_add))
print("Root Mean Squared Error (RMSE):", rmse)

# Variance and mean for reference
print("Scaled Data Variance:", np.sqrt(scaled_data.var()))
print("Scaled Data Mean:", scaled_data.mean())

# Model 2: Forecasting next year (12 months)
model_full = ExponentialSmoothing(scaled_data, trend='add', seasonal='mul', seasonal_periods=12).fit()
predictions = model_full.forecast(steps=12)

# Plot predictions
ax = scaled_data.plot(figsize=(10,5))
predictions.plot(ax=ax)
ax.legend(["Actual (Scaled)", "Forecast (Next Year)"])
ax.set_xlabel('Date')
ax.set_ylabel('Scaled Open Values')
ax.set_title('Forecasting Next Year using Exponential Smoothing')
plt.grid()
plt.show()
```
### OUTPUT:
<img width="363" height="291" alt="image" src="https://github.com/user-attachments/assets/3cccaa14-bfcd-42d6-866e-fbdf64a521d1" />

<img width="1125" height="610" alt="image" src="https://github.com/user-attachments/assets/70303fe2-78fe-43b2-81c5-debf5f4b066d" />

<img width="1130" height="617" alt="image" src="https://github.com/user-attachments/assets/a142ce4f-3ae4-4b39-bdc8-7f270d4c3ae0" />

<img width="935" height="672" alt="image" src="https://github.com/user-attachments/assets/6736580f-351a-4e01-9dad-a33574afecea" />

<img width="958" height="587" alt="image" src="https://github.com/user-attachments/assets/d9432610-e964-4efd-89f9-24b8b9ea8d82" />

### RESULT:
Thus we have successfully implemented the Moving Average Model and Exponential smoothing using python.
