import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# --- Step 1: Load and Prepare the Data ---

print("Loading and preparing data...")

# Load the training data, parsing the 'date' column as datetime objects
try:
    df = pd.read_csv('train.csv', parse_dates=['date'])
except FileNotFoundError:
    print("Error: 'train.csv' not found. Please download it and place it in the same directory.")
    exit()


df_store1_item1 = df[(df['store'] == 1) & (df['item'] == 1)]

print(f"Data for Store 1, Item 1 loaded. Shape: {df_store1_item1.shape}")

# Prepare the data for time series analysis
sales_data = df_store1_item1[['date', 'sales']].copy()
sales_data.set_index('date', inplace=True)
sales_data = sales_data.asfreq('D') # Ensure daily frequency

# --- Step 2: Exploratory Data Analysis (EDA) & Decomposition ---

print("Performing time series analysis and decomposition...")
plt.style.use('seaborn-v0_8-whitegrid')

# Plot the sales data
fig, ax = plt.subplots(figsize=(16, 6))
sales_data['sales'].plot(ax=ax)
ax.set_title('Daily Sales for Store 1, Item 1 (2013-2017)')
ax.set_xlabel('Date')
ax.set_ylabel('Sales Volume')
plt.show()

# Decompose the time series to observe trend, seasonality, and residuals
decomposition = seasonal_decompose(sales_data['sales'], model='additive', period=365)
fig = decomposition.plot()
fig.set_size_inches(14, 10)
plt.suptitle('Time Series Decomposition', y=1.02)
plt.tight_layout()
plt.show()

#
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
plot_acf(sales_data['sales'], lags=40, ax=ax1)
ax1.set_title('Autocorrelation Function (ACF)')
plot_pacf(sales_data['sales'], lags=40, ax=ax2)
ax2.set_title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

# --- Step 3: Train-Test Split and Model Training ---

print("Splitting data and training SARIMA model...")

# Split data into training and testing sets (last 90 days for testing)
train_size = len(sales_data) - 90
train, test = sales_data[0:train_size], sales_data[train_size:len(sales_data)]


my_order = (1, 1, 1)
my_seasonal_order = (1, 1, 1, 7)

# Create and fit the SARIMAX model
model = SARIMAX(train['sales'],
                order=my_order,
                seasonal_order=my_seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fitting the model
results = model.fit(disp=True) # disp=True shows convergence output

print("\n--- Model Summary ---")
print(results.summary())

# --- Step 4: Forecasting and Evaluation ---

print("Generating forecast and evaluating the model...")

# Get forecast for the test set period
pred = results.get_prediction(start=pd.to_datetime(test.index[0]), end=pd.to_datetime(test.index[-1]), dynamic=False)
pred_ci = pred.conf_int() # Get confidence intervals

# Evaluate the model
y_forecasted = pred.predicted_mean
y_truth = test['sales']
rmse = np.sqrt(mean_squared_error(y_truth, y_forecasted))
print(f'\nRoot Mean Squared Error (RMSE): {rmse:.2f}')

# Plot the forecast against the actual values
fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(sales_data.index[-200:], sales_data['sales'][-200:], label='Observed')
ax.plot(y_forecasted.index, y_forecasted.values, color='r', label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.15, label='95% Confidence Interval')

ax.set_title('Sales Forecast vs Actuals')
ax.set_xlabel('Date')
ax.set_ylabel('Sales Volume')
ax.legend()
plt.tight_layout()
plt.show()