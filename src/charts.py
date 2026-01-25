import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Dict, Any, Union

# Configure logging
logger = logging.getLogger(__name__)


def validate_dataframe(df: pd.DataFrame, required_cols: List[str]) -> bool:
    """
    Validate DataFrame has required columns.

    Args:
        df: DataFrame to validate
        required_cols: List of required column names

    Returns:
        True if valid, False otherwise
    """
    if df is None or df.empty:
        logger.warning("DataFrame is None or empty")
        return False

    # Handle multi-level columns
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    missing = [col for col in required_cols if col not in data.columns]
    if missing:
        logger.error(f"Missing required columns: {missing}")
        return False

    return True


def create_candlestick_chart(
    df: pd.DataFrame,
    title: str = "Stock Price",
    show_volume: bool = True,
    theme: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create an interactive candlestick chart with optional volume.
    
    Args:
        df: DataFrame with Date, Open, High, Low, Close, Volume columns
        title: Chart title
        show_volume: Whether to show volume subplot
        theme: Plotly theme configuration
    
    Returns:
        Plotly Figure object
    """
    if theme is None:
        theme = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'increasing_color': '#00d4aa',
            'decreasing_color': '#ef5350',
            'gridcolor': 'rgba(255,255,255,0.1)',
        }
    
    # Flatten multi-level columns if present
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Create subplots
    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
            subplot_titles=(title, 'Volume')
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data['Date'],
            open=data['Open'],
            high=data['High'],
            low=data['Low'],
            close=data['Close'],
            name='OHLC',
            increasing_line_color=theme['increasing_color'],
            decreasing_line_color=theme['decreasing_color'],
            increasing_fillcolor=theme['increasing_color'],
            decreasing_fillcolor=theme['decreasing_color'],
        ),
        row=1, col=1
    )
    
    # Volume bars
    if show_volume:
        colors = [theme['increasing_color'] if data['Close'].iloc[i] >= data['Open'].iloc[i] 
                  else theme['decreasing_color'] for i in range(len(data))]
        
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7,
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        template=theme['template'],
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600 if show_volume else 450,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    # Update axes
    fig.update_xaxes(gridcolor=theme['gridcolor'])
    fig.update_yaxes(gridcolor=theme['gridcolor'])
    
    return fig


def add_moving_averages(
    fig: go.Figure,
    df: pd.DataFrame,
    periods: List[int] = [20, 50],
    colors: Optional[List[str]] = None,
    use_subplots: bool = True
) -> go.Figure:
    """Add moving average lines to an existing chart."""
    if colors is None:
        colors = ['#667eea', '#ffa726', '#ab47bc', '#42a5f5']
    
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    for i, period in enumerate(periods):
        ma = data['Close'].rolling(window=period).mean()
        trace = go.Scatter(
            x=data['Date'],
            y=ma,
            name=f'SMA {period}',
            line=dict(color=colors[i % len(colors)], width=1.5),
            opacity=0.8,
        )
        if use_subplots:
            fig.add_trace(trace, row=1, col=1)
        else:
            fig.add_trace(trace)
    
    return fig


def add_bollinger_bands(
    fig: go.Figure,
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
    color: str = '#667eea',
    use_subplots: bool = True
) -> go.Figure:
    """Add Bollinger Bands to an existing chart."""
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    middle = data['Close'].rolling(window=period).mean()
    std = data['Close'].rolling(window=period).std()
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    # Upper band
    upper_trace = go.Scatter(
        x=data['Date'],
        y=upper,
        name='BB Upper',
        line=dict(color=color, width=1, dash='dash'),
        opacity=0.6,
    )
    
    # Lower band
    lower_trace = go.Scatter(
        x=data['Date'],
        y=lower,
        name='BB Lower',
        line=dict(color=color, width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(102, 126, 234, 0.1)',
        opacity=0.6,
    )
    
    if use_subplots:
        fig.add_trace(upper_trace, row=1, col=1)
        fig.add_trace(lower_trace, row=1, col=1)
    else:
        fig.add_trace(upper_trace)
        fig.add_trace(lower_trace)
    
    return fig


def create_indicator_chart(
    df: pd.DataFrame,
    indicator: str = 'RSI',
    title: Optional[str] = None,
    theme: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a chart for technical indicators (RSI, MACD, Stochastic).
    
    Args:
        df: DataFrame with calculated indicators
        indicator: 'RSI', 'MACD', or 'STOCH'
        title: Chart title
        theme: Plotly theme configuration
    """
    if theme is None:
        theme = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'gridcolor': 'rgba(255,255,255,0.1)',
            'colors': ['#667eea', '#00d4aa', '#ef5350'],
        }
    
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    fig = go.Figure()
    
    if indicator.upper() == 'RSI':
        if title is None:
            title = 'Relative Strength Index (RSI)'
        
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['RSI'],
                name='RSI',
                line=dict(color=theme['colors'][0], width=2),
            )
        )
        
        # Overbought/Oversold levels
        fig.add_hline(y=70, line_dash="dash", line_color=theme['colors'][2], 
                      annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color=theme['colors'][1], 
                      annotation_text="Oversold (30)")
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5)
        
        fig.update_yaxes(range=[0, 100])
    
    elif indicator.upper() == 'MACD':
        if title is None:
            title = 'MACD (Moving Average Convergence Divergence)'
        
        # MACD Line
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['MACD'],
                name='MACD',
                line=dict(color=theme['colors'][0], width=2),
            )
        )
        
        # Signal Line
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['MACD_Signal'],
                name='Signal',
                line=dict(color=theme['colors'][1], width=2),
            )
        )
        
        # Histogram
        colors = [theme['colors'][1] if val >= 0 else theme['colors'][2] 
                  for val in data['MACD_Histogram'].fillna(0)]
        
        fig.add_trace(
            go.Bar(
                x=data['Date'],
                y=data['MACD_Histogram'],
                name='Histogram',
                marker_color=colors,
                opacity=0.6,
            )
        )
        
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    elif indicator.upper() == 'STOCH':
        if title is None:
            title = 'Stochastic Oscillator'
        
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Stoch_K'],
                name='%K',
                line=dict(color=theme['colors'][0], width=2),
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=data['Stoch_D'],
                name='%D',
                line=dict(color=theme['colors'][1], width=2),
            )
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color=theme['colors'][2], 
                      annotation_text="Overbought (80)")
        fig.add_hline(y=20, line_dash="dash", line_color=theme['colors'][1], 
                      annotation_text="Oversold (20)")
        
        fig.update_yaxes(range=[0, 100])
    
    # Update layout
    fig.update_layout(
        title=title,
        template=theme['template'],
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        height=300,
        margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    fig.update_xaxes(gridcolor=theme['gridcolor'])
    fig.update_yaxes(gridcolor=theme['gridcolor'])
    
    return fig


def create_prediction_chart(
    df: pd.DataFrame,
    predictions: np.ndarray,
    future_dates: Optional[pd.DatetimeIndex] = None,
    future_predictions: Optional[np.ndarray] = None,
    confidence_interval: float = 0.05,
    title: str = "Price Prediction",
    theme: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a chart showing actual vs predicted prices with future forecast.
    
    Args:
        df: DataFrame with actual prices
        predictions: Array of predictions for historical data
        future_dates: DatetimeIndex for future predictions
        future_predictions: Array of future predictions
        confidence_interval: Confidence interval percentage
        title: Chart title
        theme: Plotly theme configuration
    """
    if theme is None:
        theme = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'gridcolor': 'rgba(255,255,255,0.1)',
            'colors': ['#667eea', '#00d4aa', '#ffa726'],
        }
    
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    fig = go.Figure()
    
    # Actual prices
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=data['Close'],
            name='Actual Price',
            line=dict(color=theme['colors'][0], width=2),
        )
    )
    
    # Historical predictions
    prediction_start = len(data) - len(predictions)
    prediction_dates = data['Date'].iloc[prediction_start:].reset_index(drop=True)
    
    fig.add_trace(
        go.Scatter(
            x=prediction_dates,
            y=predictions.flatten(),
            name='Predicted Price',
            line=dict(color=theme['colors'][1], width=2, dash='dash'),
        )
    )
    
    # Future predictions with confidence interval
    if future_dates is not None and future_predictions is not None:
        upper = future_predictions * (1 + confidence_interval)
        lower = future_predictions * (1 - confidence_interval)
        
        # Confidence interval (shaded area)
        fig.add_trace(
            go.Scatter(
                x=list(future_dates) + list(future_dates)[::-1],
                y=list(upper.flatten()) + list(lower.flatten())[::-1],
                fill='toself',
                fillcolor='rgba(255, 167, 38, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Interval',
                showlegend=True,
            )
        )
        
        # Future prediction line
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=future_predictions.flatten(),
                name='Forecast',
                line=dict(color=theme['colors'][2], width=2.5),
                mode='lines+markers',
                marker=dict(size=4),
            )
        )
        
        # Vertical line at forecast start - use add_shape to avoid Plotly annotation issues
        last_date = data['Date'].iloc[-1]
        fig.add_shape(
            type="line",
            x0=last_date,
            x1=last_date,
            y0=0,
            y1=1,
            yref="paper",
            line=dict(color="gray", width=2, dash="dash"),
        )
        # Add annotation separately
        fig.add_annotation(
            x=last_date,
            y=1,
            yref="paper",
            text="Forecast Start",
            showarrow=False,
            yshift=10,
            font=dict(color="gray")
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        template=theme['template'],
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Date",
        yaxis_title="Price (USD)",
    )
    
    fig.update_xaxes(gridcolor=theme['gridcolor'])
    fig.update_yaxes(gridcolor=theme['gridcolor'])
    
    return fig


def create_comparison_chart(
    stock_data: Dict[str, pd.DataFrame],
    normalize: bool = True,
    title: str = "Stock Comparison",
    theme: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """
    Create a multi-stock comparison chart.
    
    Args:
        stock_data: Dictionary with symbol -> DataFrame mapping
        normalize: Whether to normalize prices to percentage change
        title: Chart title
        theme: Plotly theme configuration
    """
    if theme is None:
        theme = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'gridcolor': 'rgba(255,255,255,0.1)',
            'colors': ['#667eea', '#00d4aa', '#ffa726', '#ef5350', '#ab47bc', '#42a5f5'],
        }
    
    fig = go.Figure()
    
    for i, (symbol, df) in enumerate(stock_data.items()):
        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        prices = data['Close']
        
        if normalize:
            # Normalize to percentage change from first value
            prices = (prices / prices.iloc[0] - 1) * 100
            yaxis_title = "Change (%)"
        else:
            yaxis_title = "Price (USD)"
        
        fig.add_trace(
            go.Scatter(
                x=data['Date'],
                y=prices,
                name=symbol,
                line=dict(color=theme['colors'][i % len(theme['colors'])], width=2),
            )
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        template=theme['template'],
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_title="Date",
        yaxis_title=yaxis_title,
        hovermode='x unified',
    )
    
    if normalize:
        fig.add_hline(y=0, line_dash="solid", line_color="gray", opacity=0.5)
    
    fig.update_xaxes(gridcolor=theme['gridcolor'])
    fig.update_yaxes(gridcolor=theme['gridcolor'])
    
    return fig


def create_volume_analysis_chart(
    df: pd.DataFrame,
    ma_period: int = 20,
    title: str = "Volume Analysis",
    theme: Optional[Dict[str, Any]] = None
) -> go.Figure:
    """Create a volume analysis chart with moving average."""
    if theme is None:
        theme = {
            'template': 'plotly_dark',
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'gridcolor': 'rgba(255,255,255,0.1)',
            'increasing_color': '#00d4aa',
            'decreasing_color': '#ef5350',
        }
    
    data = df.copy()
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    fig = go.Figure()
    
    # Volume bars
    colors = [theme['increasing_color'] if data['Close'].iloc[i] >= data['Open'].iloc[i] 
              else theme['decreasing_color'] for i in range(len(data))]
    
    fig.add_trace(
        go.Bar(
            x=data['Date'],
            y=data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7,
        )
    )
    
    # Volume MA
    volume_ma = data['Volume'].rolling(window=ma_period).mean()
    fig.add_trace(
        go.Scatter(
            x=data['Date'],
            y=volume_ma,
            name=f'Volume MA ({ma_period})',
            line=dict(color='#667eea', width=2),
        )
    )
    
    # Update layout
    fig.update_layout(
        title=title,
        template=theme['template'],
        paper_bgcolor=theme['paper_bgcolor'],
        plot_bgcolor=theme['plot_bgcolor'],
        height=300,
        margin=dict(l=50, r=50, t=50, b=30),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
    )
    
    fig.update_xaxes(gridcolor=theme['gridcolor'])
    fig.update_yaxes(gridcolor=theme['gridcolor'])
    
    return fig
