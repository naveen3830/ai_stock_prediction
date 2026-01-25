# indicators.py - Technical Analysis Indicators Module
"""
Technical analysis indicators for stock analysis:
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Simple Moving Average (SMA)
- Exponential Moving Average (EMA)
"""

import pandas as pd
import numpy as np


def calculate_sma(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Simple Moving Average.
    
    Args:
        data: Price series (typically closing prices)
        period: Number of periods for the moving average
    
    Returns:
        Series with SMA values
    """
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int = 20) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        data: Price series (typically closing prices)
        period: Number of periods for the moving average
    
    Returns:
        Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI oscillates between 0 and 100:
    - RSI > 70: Overbought (potential sell signal)
    - RSI < 30: Oversold (potential buy signal)
    
    Args:
        data: Price series (typically closing prices)
        period: Number of periods for RSI calculation (default: 14)
    
    Returns:
        Series with RSI values
    """
    delta = data.diff()
    
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_macd(data: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Returns MACD line, Signal line, and Histogram.
    
    Args:
        data: Price series (typically closing prices)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line EMA period (default: 9)
    
    Returns:
        Dictionary with 'macd', 'signal', and 'histogram' Series
    """
    ema_fast = calculate_ema(data, fast_period)
    ema_slow = calculate_ema(data, slow_period)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(data: pd.Series, period: int = 20, std_dev: float = 2.0) -> dict:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of:
    - Middle Band: SMA of closing prices
    - Upper Band: SMA + (std_dev * standard deviation)
    - Lower Band: SMA - (std_dev * standard deviation)
    
    Args:
        data: Price series (typically closing prices)
        period: Number of periods for SMA and std dev (default: 20)
        std_dev: Number of standard deviations (default: 2.0)
    
    Returns:
        Dictionary with 'upper', 'middle', 'lower' Series
    """
    middle = calculate_sma(data, period)
    std = data.rolling(window=period).std()
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return {
        'upper': upper,
        'middle': middle,
        'lower': lower
    }


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    
    ATR measures market volatility.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods (default: 14)
    
    Returns:
        Series with ATR values
    """
    prev_close = close.shift(1)
    
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume (OBV).
    
    OBV uses volume flow to predict price changes.
    
    Args:
        close: Close price series
        volume: Volume series
    
    Returns:
        Series with OBV values
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv = (direction * volume).cumsum()
    
    return obv


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> dict:
    """
    Calculate Stochastic Oscillator.
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        k_period: %K period (default: 14)
        d_period: %D period (default: 3)
    
    Returns:
        Dictionary with 'k' and 'd' Series
    """
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return {
        'k': stoch_k,
        'd': stoch_d
    }


def get_rsi_signal(rsi_value: float) -> str:
    """Get trading signal based on RSI value."""
    if rsi_value >= 70:
        return "Overbought"
    elif rsi_value <= 30:
        return "Oversold"
    else:
        return "Neutral"


def get_macd_signal(macd: float, signal: float) -> str:
    """Get trading signal based on MACD and signal line."""
    if macd > signal:
        return "Bullish"
    elif macd < signal:
        return "Bearish"
    else:
        return "Neutral"


def get_bollinger_signal(price: float, upper: float, lower: float) -> str:
    """Get trading signal based on Bollinger Bands."""
    if price >= upper:
        return "Overbought"
    elif price <= lower:
        return "Oversold"
    else:
        return "Neutral"


def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a DataFrame with OHLCV data.
    
    Args:
        df: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume' columns
    
    Returns:
        DataFrame with all indicators added
    """
    result = df.copy()
    
    # Flatten multi-level columns if present
    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)
    
    close = result['Close']
    high = result['High']
    low = result['Low']
    volume = result['Volume']
    
    # Moving Averages
    result['SMA_20'] = calculate_sma(close, 20)
    result['SMA_50'] = calculate_sma(close, 50)
    result['EMA_12'] = calculate_ema(close, 12)
    result['EMA_26'] = calculate_ema(close, 26)
    
    # RSI
    result['RSI'] = calculate_rsi(close)
    
    # MACD
    macd_data = calculate_macd(close)
    result['MACD'] = macd_data['macd']
    result['MACD_Signal'] = macd_data['signal']
    result['MACD_Histogram'] = macd_data['histogram']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(close)
    result['BB_Upper'] = bb_data['upper']
    result['BB_Middle'] = bb_data['middle']
    result['BB_Lower'] = bb_data['lower']
    
    # ATR
    result['ATR'] = calculate_atr(high, low, close)
    
    # OBV
    result['OBV'] = calculate_obv(close, volume)
    
    # Stochastic
    stoch_data = calculate_stochastic(high, low, close)
    result['Stoch_K'] = stoch_data['k']
    result['Stoch_D'] = stoch_data['d']
    
    return result


def get_indicator_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of the latest indicator values and signals.
    
    Args:
        df: DataFrame with calculated indicators
    
    Returns:
        Dictionary with indicator values and signals
    """
    latest = df.iloc[-1]
    
    # Get close price
    close_price = float(latest['Close'])
    
    # RSI
    rsi_value = float(latest['RSI']) if not pd.isna(latest['RSI']) else None
    rsi_signal = get_rsi_signal(rsi_value) if rsi_value else "N/A"
    
    # MACD
    macd_value = float(latest['MACD']) if not pd.isna(latest['MACD']) else None
    signal_value = float(latest['MACD_Signal']) if not pd.isna(latest['MACD_Signal']) else None
    macd_signal = get_macd_signal(macd_value, signal_value) if macd_value and signal_value else "N/A"
    
    # Bollinger Bands
    bb_upper = float(latest['BB_Upper']) if not pd.isna(latest['BB_Upper']) else None
    bb_lower = float(latest['BB_Lower']) if not pd.isna(latest['BB_Lower']) else None
    bb_signal = get_bollinger_signal(close_price, bb_upper, bb_lower) if bb_upper and bb_lower else "N/A"
    
    return {
        'rsi': {
            'value': rsi_value,
            'signal': rsi_signal
        },
        'macd': {
            'value': macd_value,
            'signal_line': signal_value,
            'signal': macd_signal
        },
        'bollinger': {
            'upper': bb_upper,
            'lower': bb_lower,
            'signal': bb_signal
        },
        'sma_20': float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
        'sma_50': float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
        'atr': float(latest['ATR']) if not pd.isna(latest['ATR']) else None,
    }
