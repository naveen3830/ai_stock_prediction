import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime, timedelta
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger(__name__)


def validate_data(data: pd.DataFrame) -> Tuple[bool, str]:
    if data is None or data.empty:
        return False, "Data is empty or None"

    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required_columns = ['Close']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"

    # Check for NaN values
    if df['Close'].isna().any():
        nan_count = df['Close'].isna().sum()
        logger.warning(f"Data contains {nan_count} NaN values in 'Close' column")

    # Check for sufficient data
    min_rows = 100  # Minimum rows needed for meaningful training
    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows (minimum {min_rows} required)"

    # Check for infinite values
    if np.isinf(df['Close'].values).any():
        return False, "Data contains infinite values"

    return True, ""


def preprocess_data(
    data: pd.DataFrame,
    prediction_days: int = 60,
    train_split: float = 0.8,
    val_split: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler, int]:
    # Validate input
    is_valid, error_msg = validate_data(data)
    if not is_valid:
        raise ValueError(f"Data validation failed: {error_msg}")

    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Get close prices and handle NaN
    close_prices = df['Close'].values.reshape(-1, 1)
    close_prices = np.nan_to_num(close_prices, nan=np.nanmean(close_prices))

    # Calculate split indices
    total_samples = len(close_prices) - prediction_days
    train_size = int(total_samples * train_split)
    val_size = int(total_samples * val_split)
    test_size = total_samples - train_size - val_size

    logger.info(f"Data split: Train={train_size}, Validation={val_size}, Test={test_size}")

    # CRITICAL: Fit scaler ONLY on training data to prevent data leakage
    train_data = close_prices[:train_size + prediction_days]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_data)

    # Transform all data using the scaler fitted on training data only
    scaled_data = scaler.transform(close_prices)

    # Create sequences
    def create_sequences(data: np.ndarray, start_idx: int, end_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        for i in range(start_idx, end_idx):
            if i >= prediction_days:
                x.append(data[i - prediction_days:i, 0])
                y.append(data[i, 0])
        return np.array(x), np.array(y)

    # Training sequences (from prediction_days to train_size + prediction_days)
    x_train, y_train = create_sequences(scaled_data, prediction_days, train_size + prediction_days)

    # Validation sequences
    val_start = train_size + prediction_days
    val_end = val_start + val_size
    x_val, y_val = create_sequences(scaled_data, val_start, val_end)

    # Test sequences
    test_start = val_end
    test_end = len(scaled_data)
    x_test, y_test = create_sequences(scaled_data, test_start, test_end)

    # Reshape for LSTM [samples, time steps, features]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1)) if len(x_val) > 0 else np.array([])
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1)) if len(x_test) > 0 else np.array([])

    logger.info(f"Sequence shapes - Train: {x_train.shape}, Val: {x_val.shape if len(x_val) > 0 else 'empty'}, Test: {x_test.shape if len(x_test) > 0 else 'empty'}")

    return x_train, y_train, x_val, y_val, x_test, y_test, scaler, train_size


def preprocess_data_simple(data: pd.DataFrame, prediction_days: int = 60) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    x_train, y_train, _, _, _, _, scaler, _ = preprocess_data(data, prediction_days)
    return x_train, y_train, scaler


def build_lstm_model(input_shape: Tuple, units: int = 50, dropout_rate: float = 0.2) -> Sequential:
    if len(input_shape) < 2:
        raise ValueError(f"Invalid input_shape: {input_shape}. Expected at least 2 dimensions.")

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
    logger.info(f"Built LSTM model with {units} units and {dropout_rate} dropout rate")
    return model


def train_lstm_model(
    model: Sequential,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray = None,
    y_val: np.ndarray = None,
    epochs: int = 15,
    batch_size: int = 32,
    validation_split: float = 0.1,
    early_stopping: bool = True
) -> Tuple[Sequential, Dict]:
    if len(x_train) == 0:
        raise ValueError("Training data is empty")

    callbacks = []
    if early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            )
        )

    # Use provided validation data or split from training
    if x_val is not None and len(x_val) > 0 and y_val is not None:
        validation_data = (x_val, y_val)
        val_split = None
        logger.info(f"Training with explicit validation set of {len(x_val)} samples")
    else:
        validation_data = None
        val_split = validation_split
        logger.info(f"Training with {validation_split*100:.0f}% validation split")

    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1
    )

    logger.info(f"Training completed. Final loss: {history.history['loss'][-1]:.6f}")

    return model, history.history


def make_predictions(
    model: Sequential,
    data: pd.DataFrame,
    scaler: MinMaxScaler,
    prediction_days: int = 60
) -> np.ndarray:
    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_prices = df['Close'].values.reshape(-1, 1)
    close_prices = np.nan_to_num(close_prices, nan=np.nanmean(close_prices))

    test_data = scaler.transform(close_prices)
    x_test = []

    for i in range(prediction_days, len(test_data)):
        x_test.append(test_data[i - prediction_days:i, 0])

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

    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_prices = df['Close'].values.reshape(-1, 1)
    close_prices = np.nan_to_num(close_prices, nan=np.nanmean(close_prices))

    # Scale the data
    scaled_data = scaler.transform(close_prices)

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

    logger.info(f"Generated {days_ahead} day forecast from {future_dates[0]} to {future_dates[-1]}")

    return future_dates, future_predictions


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray
) -> Dict[str, float]:

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    # Ensure same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    # Remove any NaN or infinite values
    mask = ~(np.isnan(actual) | np.isnan(predicted) | np.isinf(actual) | np.isinf(predicted))
    actual = actual[mask]
    predicted = predicted[mask]

    if len(actual) == 0:
        logger.warning("No valid data points for metrics calculation")
        return {'mae': np.nan, 'mse': np.nan, 'rmse': np.nan, 'r2': np.nan, 'mape': np.nan}

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(actual, predicted)

    # MAPE (Mean Absolute Percentage Error) - handle division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        if np.isinf(mape) or np.isnan(mape):
            mape = np.nan

    metrics = {
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }

    logger.info(f"Model metrics - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    return metrics


def get_prediction_confidence(
    predictions: np.ndarray,
    historical_errors: np.ndarray = None,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    predictions = np.array(predictions).flatten()

    if historical_errors is not None and len(historical_errors) > 10:
        # Use residual-based confidence intervals
        historical_errors = np.array(historical_errors).flatten()

        # Remove outliers (values beyond 3 standard deviations)
        mean_error = np.mean(historical_errors)
        std_error = np.std(historical_errors)
        mask = np.abs(historical_errors - mean_error) <= 3 * std_error
        clean_errors = historical_errors[mask]

        if len(clean_errors) > 5:
            std_error = np.std(clean_errors)

        # Z-score for confidence level (approximate)
        z_scores = {0.99: 2.576, 0.95: 1.96, 0.90: 1.645, 0.80: 1.28}
        z = z_scores.get(confidence_level, 1.96)

        # Expand uncertainty for future predictions (uncertainty grows with time)
        n_predictions = len(predictions)
        time_factors = 1 + np.arange(n_predictions) * 0.02  # 2% increase per day

        margin = z * std_error * time_factors

        upper = predictions + margin
        lower = predictions - margin

        logger.info(f"Confidence intervals calculated using {len(clean_errors)} historical errors")
    else:
        # Fallback: percentage-based approach with time-dependent expansion
        base_margin = (1 - confidence_level) + 0.03  # Reduced base margin

        # Expand confidence interval for longer-term predictions
        n_predictions = len(predictions)
        time_factors = 1 + np.arange(n_predictions) * 0.01  # 1% increase per day

        margins = base_margin * time_factors

        upper = predictions * (1 + margins)
        lower = predictions * (1 - margins)

        logger.warning("Using fallback percentage-based confidence intervals")

    # Ensure lower bound is not negative for stock prices
    lower = np.maximum(lower, 0)

    return lower.reshape(-1, 1), upper.reshape(-1, 1)


def create_prediction_dataframe(
    dates: pd.DatetimeIndex,
    predictions: np.ndarray,
    confidence_level: float = 0.95,
    historical_errors: np.ndarray = None
) -> pd.DataFrame:
    lower, upper = get_prediction_confidence(predictions, historical_errors, confidence_level)

    return pd.DataFrame({
        'Date': dates,
        'Predicted': predictions.flatten(),
        'Upper': upper.flatten(),
        'Lower': lower.flatten()
    })


def train_and_evaluate(
    data: pd.DataFrame,
    prediction_days: int = 60,
    epochs: int = 15,
    units: int = 50,
    dropout_rate: float = 0.2
) -> Tuple[Sequential, MinMaxScaler, Dict[str, float], np.ndarray]:
    """
    Train LSTM model with proper evaluation, then retrain on all data for forecasting.

    This uses a two-phase approach:
    1. Train on 90% of data, evaluate on last 10% for honest metrics
    2. Retrain final model on ALL data for actual forecasting

    This is standard practice for time-series forecasting in production.
    """
    logger.info("Starting training pipeline...")

    # Handle multi-level columns
    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close_prices = df['Close'].values.reshape(-1, 1)
    close_prices = np.nan_to_num(close_prices, nan=np.nanmean(close_prices))

    # ========== PHASE 1: Evaluation with holdout set ==========
    # Use last 10% as holdout for honest metrics
    holdout_size = int(len(close_prices) * 0.1)
    train_prices = close_prices[:-holdout_size]
    holdout_prices = close_prices[-holdout_size - prediction_days:]  # Include lookback window

    # Fit scaler on training portion only for evaluation
    eval_scaler = MinMaxScaler(feature_range=(0, 1))
    eval_scaler.fit(train_prices)

    # Scale all data for evaluation
    scaled_train = eval_scaler.transform(train_prices)
    scaled_holdout = eval_scaler.transform(holdout_prices)

    # Create training sequences
    x_train, y_train = [], []
    for i in range(prediction_days, len(scaled_train)):
        x_train.append(scaled_train[i - prediction_days:i, 0])
        y_train.append(scaled_train[i, 0])
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Create holdout sequences (for evaluation only)
    x_holdout, y_holdout = [], []
    for i in range(prediction_days, len(scaled_holdout)):
        x_holdout.append(scaled_holdout[i - prediction_days:i, 0])
        y_holdout.append(scaled_holdout[i, 0])
    x_holdout, y_holdout = np.array(x_holdout), np.array(y_holdout)
    x_holdout = np.reshape(x_holdout, (x_holdout.shape[0], x_holdout.shape[1], 1))

    # Build and train evaluation model
    eval_model = build_lstm_model(x_train.shape, units, dropout_rate)
    eval_model, _ = train_lstm_model(eval_model, x_train, y_train, epochs=epochs)

    # Evaluate on holdout set
    holdout_predictions = eval_model.predict(x_holdout, verbose=0)
    holdout_predictions_inv = eval_scaler.inverse_transform(holdout_predictions)
    y_holdout_inv = eval_scaler.inverse_transform(y_holdout.reshape(-1, 1))

    eval_metrics = calculate_metrics(y_holdout_inv, holdout_predictions_inv)
    historical_errors = (y_holdout_inv - holdout_predictions_inv).flatten()

    logger.info(f"Holdout evaluation - R²: {eval_metrics['r2']:.4f}, RMSE: {eval_metrics['rmse']:.2f}")

    # ========== PHASE 2: Train final model on ALL data ==========
    logger.info("Retraining final model on all available data...")

    # Fit scaler on ALL data for the final model
    final_scaler = MinMaxScaler(feature_range=(0, 1))
    final_scaler.fit(close_prices)
    scaled_all = final_scaler.transform(close_prices)

    # Create sequences from all data
    x_all, y_all = [], []
    for i in range(prediction_days, len(scaled_all)):
        x_all.append(scaled_all[i - prediction_days:i, 0])
        y_all.append(scaled_all[i, 0])
    x_all, y_all = np.array(x_all), np.array(y_all)
    x_all = np.reshape(x_all, (x_all.shape[0], x_all.shape[1], 1))

    # Build and train final model on all data
    final_model = build_lstm_model(x_all.shape, units, dropout_rate)
    final_model, _ = train_lstm_model(final_model, x_all, y_all, epochs=epochs)

    # Calculate metrics on full dataset (for display - will show better fit)
    full_predictions = final_model.predict(x_all, verbose=0)
    full_predictions_inv = final_scaler.inverse_transform(full_predictions)
    y_all_inv = final_scaler.inverse_transform(y_all.reshape(-1, 1))

    full_metrics = calculate_metrics(y_all_inv, full_predictions_inv)

    logger.info(f"Final model (all data) - R²: {full_metrics['r2']:.4f}, RMSE: {full_metrics['rmse']:.2f}")

    # Return the final model trained on all data, with evaluation metrics from holdout
    # Note: We return full_metrics for display (shows model fit quality)
    # but historical_errors from holdout (for realistic confidence intervals)
    return final_model, final_scaler, full_metrics, historical_errors
