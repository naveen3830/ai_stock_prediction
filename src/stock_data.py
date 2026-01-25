# stock_data.py - Extended Stock Data Module
"""
Stock data management with:
- Extended stock mapping with sectors
- Search functionality
- Multi-stock comparison support
- Watchlist management
- Comprehensive error handling and logging
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Optional, Union
from datetime import date
import streamlit as st

# Configure logging
logger = logging.getLogger(__name__)


# Extended stock mapping with sector information
STOCK_DATABASE = {
    # Technology
    "AAPL": {"name": "Apple Inc.", "sector": "Technology"},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology"},
    "GOOGL": {"name": "Alphabet Inc. Class A", "sector": "Technology"},
    "GOOG": {"name": "Alphabet Inc. Class C", "sector": "Technology"},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Technology"},
    "META": {"name": "Meta Platforms Inc.", "sector": "Technology"},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology"},
    "TSLA": {"name": "Tesla Inc.", "sector": "Technology"},
    "AMD": {"name": "Advanced Micro Devices", "sector": "Technology"},
    "INTC": {"name": "Intel Corporation", "sector": "Technology"},
    "ADBE": {"name": "Adobe Inc.", "sector": "Technology"},
    "CRM": {"name": "Salesforce Inc.", "sector": "Technology"},
    "CSCO": {"name": "Cisco Systems Inc.", "sector": "Technology"},
    "AVGO": {"name": "Broadcom Inc.", "sector": "Technology"},
    "TXN": {"name": "Texas Instruments", "sector": "Technology"},
    "ORCL": {"name": "Oracle Corporation", "sector": "Technology"},
    "QCOM": {"name": "Qualcomm Inc.", "sector": "Technology"},
    "IBM": {"name": "IBM Corporation", "sector": "Technology"},
    
    # Finance
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Finance"},
    "BAC": {"name": "Bank of America Corp.", "sector": "Finance"},
    "WFC": {"name": "Wells Fargo & Company", "sector": "Finance"},
    "GS": {"name": "Goldman Sachs Group Inc.", "sector": "Finance"},
    "MS": {"name": "Morgan Stanley", "sector": "Finance"},
    "C": {"name": "Citigroup Inc.", "sector": "Finance"},
    "SCHW": {"name": "Charles Schwab Corp.", "sector": "Finance"},
    "BLK": {"name": "BlackRock Inc.", "sector": "Finance"},
    "V": {"name": "Visa Inc.", "sector": "Finance"},
    "MA": {"name": "Mastercard Inc.", "sector": "Finance"},
    "AXP": {"name": "American Express Co.", "sector": "Finance"},
    "BRK.B": {"name": "Berkshire Hathaway Inc.", "sector": "Finance"},
    
    # Healthcare
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare"},
    "UNH": {"name": "UnitedHealth Group Inc.", "sector": "Healthcare"},
    "PFE": {"name": "Pfizer Inc.", "sector": "Healthcare"},
    "ABBV": {"name": "AbbVie Inc.", "sector": "Healthcare"},
    "MRK": {"name": "Merck & Co. Inc.", "sector": "Healthcare"},
    "TMO": {"name": "Thermo Fisher Scientific", "sector": "Healthcare"},
    "ABT": {"name": "Abbott Laboratories", "sector": "Healthcare"},
    "LLY": {"name": "Eli Lilly and Company", "sector": "Healthcare"},
    "BMY": {"name": "Bristol-Myers Squibb", "sector": "Healthcare"},
    "AMGN": {"name": "Amgen Inc.", "sector": "Healthcare"},
    
    # Consumer
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer"},
    "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer"},
    "KO": {"name": "Coca-Cola Company", "sector": "Consumer"},
    "PEP": {"name": "PepsiCo Inc.", "sector": "Consumer"},
    "COST": {"name": "Costco Wholesale Corp.", "sector": "Consumer"},
    "HD": {"name": "Home Depot Inc.", "sector": "Consumer"},
    "NKE": {"name": "Nike Inc.", "sector": "Consumer"},
    "MCD": {"name": "McDonald's Corporation", "sector": "Consumer"},
    "SBUX": {"name": "Starbucks Corporation", "sector": "Consumer"},
    "DIS": {"name": "Walt Disney Company", "sector": "Consumer"},
    "CMCSA": {"name": "Comcast Corporation", "sector": "Consumer"},
    
    # Energy
    "XOM": {"name": "Exxon Mobil Corporation", "sector": "Energy"},
    "CVX": {"name": "Chevron Corporation", "sector": "Energy"},
    "COP": {"name": "ConocoPhillips", "sector": "Energy"},
    "SLB": {"name": "Schlumberger Limited", "sector": "Energy"},
    "EOG": {"name": "EOG Resources Inc.", "sector": "Energy"},
    
    # Industrial
    "BA": {"name": "Boeing Company", "sector": "Industrial"},
    "HON": {"name": "Honeywell International", "sector": "Industrial"},
    "UPS": {"name": "United Parcel Service", "sector": "Industrial"},
    "CAT": {"name": "Caterpillar Inc.", "sector": "Industrial"},
    "GE": {"name": "General Electric Co.", "sector": "Industrial"},
    "MMM": {"name": "3M Company", "sector": "Industrial"},
    "LMT": {"name": "Lockheed Martin Corp.", "sector": "Industrial"},
    "RTX": {"name": "Raytheon Technologies", "sector": "Industrial"},
    
    # Telecom
    "VZ": {"name": "Verizon Communications", "sector": "Telecom"},
    "T": {"name": "AT&T Inc.", "sector": "Telecom"},
    "TMUS": {"name": "T-Mobile US Inc.", "sector": "Telecom"},
    
    # Real Estate
    "AMT": {"name": "American Tower Corp.", "sector": "Real Estate"},
    "PLD": {"name": "Prologis Inc.", "sector": "Real Estate"},
    "CCI": {"name": "Crown Castle Inc.", "sector": "Real Estate"},
    
    # Utilities
    "NEE": {"name": "NextEra Energy Inc.", "sector": "Utilities"},
    "DUK": {"name": "Duke Energy Corp.", "sector": "Utilities"},
    "SO": {"name": "Southern Company", "sector": "Utilities"},
    
    # Materials
    "LIN": {"name": "Linde plc", "sector": "Materials"},
    "APD": {"name": "Air Products & Chemicals", "sector": "Materials"},
    "SHW": {"name": "Sherwin-Williams Co.", "sector": "Materials"},
}


def get_all_sectors() -> List[str]:
    """Get list of all unique sectors."""
    sectors = set(stock["sector"] for stock in STOCK_DATABASE.values())
    return sorted(list(sectors))


def get_stocks_by_sector(sector: str = None) -> Dict[str, Dict]:
    """
    Get stocks filtered by sector.
    
    Args:
        sector: Sector name to filter by. If None, returns all stocks.
    
    Returns:
        Dictionary of symbol -> stock info
    """
    if sector is None or sector == "All":
        return STOCK_DATABASE
    
    return {
        symbol: info 
        for symbol, info in STOCK_DATABASE.items() 
        if info["sector"] == sector
    }


def search_stocks(query: str) -> Dict[str, Dict]:
    """
    Search stocks by name or symbol.
    
    Args:
        query: Search query string
    
    Returns:
        Dictionary of matching symbol -> stock info
    """
    query_lower = query.lower()
    results = {}
    
    for symbol, info in STOCK_DATABASE.items():
        if (query_lower in symbol.lower() or 
            query_lower in info["name"].lower()):
            results[symbol] = info
    
    return results


def get_stock_info(symbol: str) -> Optional[Dict]:
    """Get info for a specific stock symbol."""
    return STOCK_DATABASE.get(symbol)


def get_stock_display_name(symbol: str) -> str:
    """Get display name for a stock (Name (SYMBOL))."""
    info = STOCK_DATABASE.get(symbol)
    if info:
        return f"{info['name']} ({symbol})"
    return symbol


def get_symbol_options() -> List[str]:
    """Get list of stock display names for selection."""
    return [get_stock_display_name(symbol) for symbol in sorted(STOCK_DATABASE.keys())]


def extract_symbol_from_display(display_name: str) -> str:
    """Extract symbol from display name format 'Name (SYMBOL)'."""
    if "(" in display_name and ")" in display_name:
        return display_name.split("(")[-1].replace(")", "").strip()
    return display_name


def validate_symbol(symbol: str) -> bool:
    """
    Validate stock symbol format.

    Args:
        symbol: Stock ticker symbol to validate

    Returns:
        True if valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False

    # Basic validation: alphanumeric with optional dots (e.g., BRK.B)
    cleaned = symbol.replace(".", "").replace("-", "")
    if not cleaned.isalnum():
        return False

    if len(symbol) > 10:  # Most symbols are under 10 characters
        return False

    return True


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(symbol: str, start_date: str = "2015-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Load stock data from Yahoo Finance with comprehensive error handling.

    Args:
        symbol: Stock ticker symbol
        start_date: Start date for data
        end_date: End date for data (defaults to today)

    Returns:
        DataFrame with stock data, or empty DataFrame on error
    """
    # Validate inputs
    if not validate_symbol(symbol):
        logger.error(f"Invalid stock symbol: {symbol}")
        return pd.DataFrame()

    if end_date is None:
        end_date = date.today().strftime("%Y-%m-%d")

    try:
        logger.info(f"Loading stock data for {symbol} from {start_date} to {end_date}")
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)

        if data.empty:
            logger.warning(f"No data returned for {symbol}")
            return pd.DataFrame()

        data.reset_index(inplace=True)

        # Validate data quality
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

        # Handle multi-level columns from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        missing_cols = [col for col in required_columns if col not in data.columns]
        if missing_cols:
            logger.error(f"Missing columns in data for {symbol}: {missing_cols}")
            return pd.DataFrame()

        # Check for excessive NaN values
        nan_pct = data['Close'].isna().sum() / len(data) * 100
        if nan_pct > 10:
            logger.warning(f"High NaN percentage ({nan_pct:.1f}%) in {symbol} data")

        # Fill minor gaps with forward fill
        data = data.ffill().bfill()

        logger.info(f"Successfully loaded {len(data)} rows for {symbol}")
        return data

    except Exception as e:
        logger.error(f"Error loading data for {symbol}: {str(e)}")
        return pd.DataFrame()


def load_multiple_stocks(symbols: List[str], start_date: str = "2015-01-01", end_date: str = None) -> Dict[str, pd.DataFrame]:
    """
    Load data for multiple stocks with error handling.

    Args:
        symbols: List of stock ticker symbols
        start_date: Start date for data
        end_date: End date for data

    Returns:
        Dictionary of symbol -> DataFrame (only includes successful loads)
    """
    if not symbols:
        logger.warning("No symbols provided to load_multiple_stocks")
        return {}

    result = {}
    failed_symbols = []

    for symbol in symbols:
        try:
            data = load_stock_data(symbol, start_date, end_date)
            if not data.empty:
                result[symbol] = data
            else:
                failed_symbols.append(symbol)
        except Exception as e:
            logger.error(f"Error loading {symbol}: {str(e)}")
            failed_symbols.append(symbol)

    if failed_symbols:
        logger.warning(f"Failed to load data for: {', '.join(failed_symbols)}")

    logger.info(f"Successfully loaded {len(result)}/{len(symbols)} stocks")
    return result


def get_current_price_info(data: pd.DataFrame) -> Optional[Dict]:
    """
    Get current price information from stock data with error handling.

    Args:
        data: DataFrame with stock price data

    Returns:
        Dictionary with current price, change, and change percentage, or None on error
    """
    if data is None or data.empty or len(data) < 2:
        logger.warning("Insufficient data for price info calculation")
        return None

    try:
        # Flatten columns if multi-level
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Validate required columns exist
        required_cols = ['Close', 'Open', 'High', 'Low', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logger.error(f"Missing columns for price info: {missing}")
            return None

        current = float(df['Close'].iloc[-1])
        previous = float(df['Close'].iloc[-2])

        # Handle edge case of zero previous price
        if previous == 0:
            logger.warning("Previous close is zero, cannot calculate change percentage")
            change_pct = 0.0
        else:
            change_pct = ((current - previous) / previous) * 100

        change = current - previous

        # Get additional stats with fallbacks
        tail_252 = df.tail(252)
        high_52w = float(tail_252['High'].max()) if len(tail_252) > 0 else current
        low_52w = float(tail_252['Low'].min()) if len(tail_252) > 0 else current

        tail_20 = df.tail(20)
        avg_volume = float(tail_20['Volume'].mean()) if len(tail_20) > 0 else 0

        return {
            'current_price': current,
            'previous_close': previous,
            'change': change,
            'change_pct': change_pct,
            'high_52w': high_52w,
            'low_52w': low_52w,
            'avg_volume': avg_volume,
            'open': float(df['Open'].iloc[-1]),
            'high': float(df['High'].iloc[-1]),
            'low': float(df['Low'].iloc[-1]),
            'volume': float(df['Volume'].iloc[-1]),
        }

    except Exception as e:
        logger.error(f"Error calculating price info: {str(e)}")
        return None


def calculate_returns(data: pd.DataFrame, periods: List[int] = [5, 20, 60, 252]) -> Dict[str, float]:
    """
    Calculate returns over various periods with error handling.

    Args:
        data: Stock price DataFrame
        periods: List of periods (in days) to calculate returns

    Returns:
        Dictionary with period -> return percentage
    """
    if data is None or data.empty:
        logger.warning("Empty data provided for returns calculation")
        return {}

    try:
        df = data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if 'Close' not in df.columns:
            logger.error("'Close' column not found in data")
            return {}

        returns = {}
        current_price = float(df['Close'].iloc[-1])

        period_names = {
            5: '1W',
            20: '1M',
            60: '3M',
            252: '1Y'
        }

        for period in periods:
            if len(df) > period:
                past_price = float(df['Close'].iloc[-(period+1)])

                # Handle edge case of zero past price
                if past_price == 0:
                    logger.warning(f"Past price is zero for {period}-day return calculation")
                    continue

                ret = ((current_price - past_price) / past_price) * 100
                name = period_names.get(period, f'{period}D')
                returns[name] = ret

        return returns

    except Exception as e:
        logger.error(f"Error calculating returns: {str(e)}")
        return {}


# Watchlist management using session state
def init_watchlist():
    """Initialize watchlist in session state."""
    if 'watchlist' not in st.session_state:
        st.session_state.watchlist = []


def add_to_watchlist(symbol: str):
    """Add a stock to the watchlist."""
    init_watchlist()
    if symbol not in st.session_state.watchlist:
        st.session_state.watchlist.append(symbol)


def remove_from_watchlist(symbol: str):
    """Remove a stock from the watchlist."""
    init_watchlist()
    if symbol in st.session_state.watchlist:
        st.session_state.watchlist.remove(symbol)


def get_watchlist() -> List[str]:
    """Get the current watchlist."""
    init_watchlist()
    return st.session_state.watchlist


def is_in_watchlist(symbol: str) -> bool:
    """Check if a stock is in the watchlist."""
    init_watchlist()
    return symbol in st.session_state.watchlist
