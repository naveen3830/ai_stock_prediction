import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
import itertools
from sklearn.metrics import mean_squared_error
from datetime import datetime
from statsmodels.tsa.stattools import adfuller


warnings.filterwarnings("ignore")  # To ignore warnings from ARIMA


# Define parameter range
p = d = q = 1  # This gives a range of (0, 1, 2)
param_combinations = list(itertools.product(p, q))


def preprocess_data(data, start_date, end_date):
    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])


    # Limit the data to a specific time range if provided
    if start_date and end_date:
        mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)
        filtered_data = data.loc[mask]
    else:
        filtered_data = data


    # Use the 'Close' prices for analysis from the filtered data
    close_prices = filtered_data['Close']


    # Split into training and testing datasets
    train_size = int(len(close_prices) * 0.8)
    train_data = close_prices[:train_size]
    test_data = close_prices[train_size:]
   
    return close_prices, train_data, test_data


# Function to apply manual differencing
def manual_differencing(series, order=1):
    diff_series = series.diff(periods=order).dropna()
    return diff_series


# Function to determine the best differencing order
def find_best_differencing(series, max_order=3):
    for order in range(1, max_order + 1):
        diff_series = manual_differencing(series, order=order)
        adf_result = adfuller(diff_series)
        p_value = adf_result[1]
       
        # If the p-value is less than 0.05, the series is stationary
        if p_value < 0.05:
            print(f"Stationarity achieved with differencing order: {order}")
            return order
   
    # If no order results in stationarity, return the max_order differencing
    print(f"Stationarity not achieved up to differencing order: {max_order}. Returning maximum differencing.")
    return max_order, manual_differencing(series, order=max_order)


def build_arima_model(train_data, test_data):
    # Find the best differencing order for the training data
    best_d= find_best_differencing(train_data)
   
    best_aic = float('inf')
    best_order = None
    best_model = None


    for order in param_combinations:
        try:
            # Use the best differencing order (d) found
            model_order = (order[0], best_d, order[1])
            model = ARIMA(train_data, order=model_order)
            model_fit = model.fit()
            aic = model_fit.aic
           
            if aic < best_aic:
                best_aic = aic
                best_order = model_order
                best_model = model_fit
               
                # Apply walk-forward validation
                print(f'Performing walk-forward validation for order: {model_order}')
                walk_forward_validation(train_data, test_data, model_order)
        except Exception as e:
            continue


    print(f'Best Order: {best_order}, AIC: {best_aic}')
    print(best_model.summary())
    return best_model


def calculate_residuals(model_fit):
    # Get residuals from the ARIMA model
    residuals = model_fit.resid


    # Convert residuals to a DataFrame
    residuals_df = pd.DataFrame(residuals, columns=['Residuals'])
    return residuals_df


def walk_forward_validation(train_data, test_data, order):
    history = [x for x in train_data]
    predictions = list()


    # Walk-forward validation
    for t in range(len(test_data)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test_data[t]
        history.append(obs)
        print('predicted=%f, expected=%f' % (yhat, obs))


    # Evaluate forecasts
    rmse = sqrt(mean_squared_error(test_data, predictions))
    print('Test RMSE: %.3f' % rmse)


def make_predictions(model_fit, test_data):
    predictions = model_fit.forecast(steps=len(test_data))
    #print(predictions)
    return predictions