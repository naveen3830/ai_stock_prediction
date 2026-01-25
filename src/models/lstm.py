# lstm_model.py - Enhanced LSTM Model for Stock Prediction
"""
Enhanced LSTM model with:
- Dropout layers for regularization
- Future forecasting capabilities
- Model performance metrics
- Caching support for Streamlit
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def preprocess_data(data: pd.DataFrame, prediction_days: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Preprocess stock data for LSTM model.
    
    Args:
        data: DataFrame with 'Close' column
        prediction_days: Number of lookback days for sequences
    
    Returns:
        Tuple of (x_train, y_train, scaler)
    """
    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

    # Prepare the training data
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler


def build_lstm_model(input_shape: Tuple, units: int = 50, dropout_rate: float = 0.2) -> Sequential:
    """
    Build an enhanced LSTM model with dropout for regularization.
    
    Args:
        input_shape: Shape of input data
        units: Number of LSTM units
        dropout_rate: Dropout rate for regularization
    
    Returns:
        Compiled Keras Sequential model
    """
    model = Sequential([
        LSTM(units=units, return_sequences=True, input_shape=(input_shape[1], 1)),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=True),
        Dropout(dropout_rate),
        LSTM(units=units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(units=units, activation='relu'),
        Dense(units=25, activation='relu'),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model


def train_lstm_model(
    model: Sequential, 
    x_train: np.ndarray, 
    y_train: np.ndarray, 
    epochs: int = 10, 
    batch_size: int = 32,
    validation_split: float = 0.1,
    early_stopping: bool = True
) -> Tuple[Sequential, Dict]:
    """
    Train the LSTM model with optional early stopping.
    
    Args:
        model: Keras model to train
        x_train: Training features
        y_train: Training targets
        epochs: Number of training epochs
        batch_size: Training batch size
        validation_split: Fraction of data for validation
        early_stopping: Whether to use early stopping
    
    Returns:
        Tuple of (trained model, training history)
    """
    callbacks = []
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=0
            )
        )
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0
    )
    
    return model, history.history


def make_predictions(
    model: Sequential, 
    data: pd.DataFrame, 
    scaler: MinMaxScaler, 
    prediction_days: int = 60
) -> np.ndarray:
    """
    Make predictions on historical data.
    
    Args:
        model: Trained Keras model
        data: DataFrame with price data
        scaler: Fitted MinMaxScaler
        prediction_days: Number of lookback days
    
    Returns:
        Array of predictions
    """
    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    test_data = scaler.transform(df['Close'].values.reshape(-1, 1))
    x_test = [test_data[i - prediction_days:i, 0] for i in range(prediction_days, len(test_data))]
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    predictions = model.predict(x_test, verbose=0)
    predictions = scaler.inverse_transform(predictions)
    return predictions


def forecast_future(
    model: Sequential,
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    days_ahead: int = 30,
    prediction_days: int = 60
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Forecast future stock prices.
    
    Uses iterative prediction where each predicted value is used
    as input for the next prediction.
    
    Args:
        model: Trained Keras model
        data: DataFrame with historical price data
        scaler: Fitted MinMaxScaler
        days_ahead: Number of days to forecast
        prediction_days: Number of lookback days
    
    Returns:
        Tuple of (future_dates, predictions)
    """
    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    # Scale the data
    scaled_data = scaler.transform(df['Close'].values.reshape(-1, 1))
    
    # Get the last sequence
    last_sequence = scaled_data[-prediction_days:].flatten()
    
    # Generate future predictions
    future_predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Prepare input
        x_input = current_sequence.reshape(1, prediction_days, 1)
        
        # Predict next value
        pred = model.predict(x_input, verbose=0)[0, 0]
        future_predictions.append(pred)
        
        # Update sequence
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[-1] = pred
    
    # Inverse transform predictions
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = scaler.inverse_transform(future_predictions)
    
    # Generate future dates
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=days_ahead,
        freq='B'  # Business days
    )
    
    return future_dates, future_predictions


def calculate_metrics(
    actual: np.ndarray, 
    predicted: np.ndarray
) -> Dict[str, float]:
    """
    Calculate model performance metrics.
    
    Args:
        actual: Array of actual values
        predicted: Array of predicted values
    
    Returns:
        Dictionary of metrics (MAE, MSE, RMSE, R2, MAPE)
    """
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()
    
    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


def get_prediction_confidence(
    predictions: np.ndarray,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate confidence intervals for predictions.
    
    Uses a simple approach based on percentage deviation.
    For more accurate intervals, use ensemble methods or
    Monte Carlo dropout in production.
    
    Args:
        predictions: Array of predictions
        confidence_level: Confidence level (0.0 to 1.0)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    # Use a simple percentage-based approach
    # The margin increases for lower confidence levels
    margin = (1 - confidence_level) + 0.05
    
    upper = predictions * (1 + margin)
    lower = predictions * (1 - margin)
    
    return lower, upper


def create_prediction_dataframe(
    dates: pd.DatetimeIndex,
    predictions: np.ndarray,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Create a DataFrame with predictions and confidence intervals.
    
    Args:
        dates: DatetimeIndex for predictions
        predictions: Array of predicted values
        confidence_level: Confidence level for intervals
    
    Returns:
        DataFrame with Date, Predicted, Upper, Lower columns
    """
    lower, upper = get_prediction_confidence(predictions, confidence_level)
    
    return pd.DataFrame({
        'Date': dates,
        'Predicted': predictions.flatten(),
        'Upper': upper.flatten(),
        'Lower': lower.flatten()
    })
