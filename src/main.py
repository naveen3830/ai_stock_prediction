"""
AI Stock Predictor
==================
A modern, feature-rich stock analysis and prediction application
with interactive charts, technical indicators, and AI-powered forecasting.

DISCLAIMER: This application is for educational purposes only.
It is not financial advice. Always consult a qualified financial
advisor before making investment decisions.
"""

import streamlit as st
from datetime import date
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

# Configure logging first
from logging_config import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Import custom modules
from styles import get_theme_css, render_header, render_stock_badge, render_price_display, render_indicator_signal, render_section_header, get_plotly_theme
from charts import create_candlestick_chart, create_indicator_chart, create_prediction_chart, create_comparison_chart, add_moving_averages, add_bollinger_bands, create_volume_analysis_chart
from stock_data import STOCK_DATABASE, get_all_sectors, get_stocks_by_sector, get_stock_info, load_stock_data, load_multiple_stocks, get_current_price_info, calculate_returns, add_to_watchlist, remove_from_watchlist, get_watchlist, is_in_watchlist, extract_symbol_from_display
from models.indicators import calculate_all_indicators, get_indicator_summary
from models import lstm

logger.info("Application started")

# Constants
START_DATE = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

# Page Configuration
st.set_page_config(
    page_title="AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'selected_symbol' not in st.session_state:
    st.session_state.selected_symbol = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Apply dark theme (always)
st.markdown(get_theme_css(True), unsafe_allow_html=True)
plotly_theme = get_plotly_theme(True)


# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("## Stock Selection")
    
    # Sector Filter
    sectors = ["All"] + get_all_sectors()
    selected_sector = st.selectbox("Filter by Sector", sectors)
    
    # Get filtered stocks
    filtered_stocks = get_stocks_by_sector(None if selected_sector == "All" else selected_sector)
    
    # Stock selector
    stock_options = [f"{info['name']} ({symbol})" for symbol, info in sorted(filtered_stocks.items())]
    
    if stock_options:
        selected_stock = st.selectbox(
            "Select Stock",
            options=[""] + stock_options,
            help="Choose a stock to analyze"
        )
        
        if selected_stock:
            st.session_state.selected_symbol = extract_symbol_from_display(selected_stock)
    else:
        st.warning("No stocks found matching your criteria")
    
    st.markdown("---")
    
    # Watchlist
    st.markdown("## Watchlist")
    watchlist = get_watchlist()
    
    if watchlist:
        for symbol in watchlist:
            col1, col2 = st.columns([4, 1])
            with col1:
                info = get_stock_info(symbol)
                if info:
                    st.markdown(f"**{symbol}** - {info['name'][:20]}...")
            with col2:
                if st.button("✕", key=f"remove_{symbol}"):
                    remove_from_watchlist(symbol)
                    st.rerun()
    else:
        st.caption("No stocks in watchlist")


# ============== MAIN CONTENT ==============
# Header
st.markdown(
    render_header(
        "AI Stock Predictor",
        "Advanced Stock Analysis & Forecasting"
    ),
    unsafe_allow_html=True
)

# Check if stock is selected
if st.session_state.selected_symbol:
    symbol = st.session_state.selected_symbol
    stock_info = get_stock_info(symbol)
    
    # Stock Badge and Watchlist Button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(
            render_stock_badge(symbol, stock_info['name'] if stock_info else symbol),
            unsafe_allow_html=True
        )
    with col2:
        if is_in_watchlist(symbol):
            if st.button("⭐ Remove from Watchlist"):
                remove_from_watchlist(symbol)
                st.rerun()
        else:
            if st.button("☆ Add to Watchlist"):
                add_to_watchlist(symbol)
                st.rerun()
    
    # Load Data
    with st.spinner(f"Loading data for {symbol}..."):
        data = load_stock_data(symbol, START_DATE, TODAY)
    
    if data.empty:
        st.error(f"Could not load data for {symbol}")
    else:
        # Calculate indicators
        data_with_indicators = calculate_all_indicators(data)
        price_info = get_current_price_info(data)
        returns = calculate_returns(data)
        indicator_summary = get_indicator_summary(data_with_indicators)
        
        # Price Display
        if price_info:
            st.markdown(
                render_price_display(
                    price_info['current_price'],
                    price_info['change'],
                    price_info['change_pct']
                ),
                unsafe_allow_html=True
            )
        
        # Quick Stats Cards - Using Streamlit native metrics
        st.markdown(render_section_header("Key Metrics", ""), unsafe_allow_html=True)
        
        # Date range info
        df_temp = data.copy()
        if isinstance(df_temp.columns, pd.MultiIndex):
            df_temp.columns = df_temp.columns.get_level_values(0)
        st.caption(f"Data range: {df_temp['Date'].iloc[0].strftime('%b %d, %Y')} to {df_temp['Date'].iloc[-1].strftime('%b %d, %Y')} ({len(df_temp)} trading days)")
        
        # First row of metrics
        if price_info:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("52W High", f"${price_info['high_52w']:,.2f}")
            with col2:
                st.metric("52W Low", f"${price_info['low_52w']:,.2f}")
            with col3:
                st.metric("Avg Volume", f"{price_info['avg_volume']/1e6:.1f}M")
            with col4:
                st.metric("Today's Volume", f"{price_info['volume']/1e6:.1f}M")
        
        # Returns row
        if returns:
            ret_cols = st.columns(len(returns))
            for i, (period, ret) in enumerate(returns.items()):
                with ret_cols[i]:
                    delta_str = f"{ret:+.1f}%"
                    st.metric(f"{period} Return", f"{ret:+.1f}%", delta=delta_str)
        
        # Main Tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Overview",
            "Technical Analysis",
            "AI Forecast",
            "Compare Stocks",
            "Export Data"
        ])
        
        # ============== TAB 1: OVERVIEW ==============
        with tab1:
            st.markdown(render_section_header("Price Chart", "📈"), unsafe_allow_html=True)
            
            # Chart options
            chart_col1, chart_col2, chart_col3 = st.columns([2, 2, 2])
            with chart_col1:
                chart_type = st.selectbox("Chart Type", ["Candlestick", "Line"])
            with chart_col2:
                show_volume = st.checkbox("Show Volume", value=True)
            with chart_col3:
                show_ma = st.checkbox("Show Moving Averages", value=True)
            
            # Create chart
            if chart_type == "Candlestick":
                fig = create_candlestick_chart(data, f"{symbol} Price Chart", show_volume, plotly_theme)
            else:
                import plotly.graph_objects as go
                fig = go.Figure()
                df = data.copy()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close', line=dict(color=plotly_theme['colors'][0])))
                fig.update_layout(
                    title=f"{symbol} Price Chart",
                    template=plotly_theme['template'],
                    paper_bgcolor=plotly_theme['paper_bgcolor'],
                    plot_bgcolor=plotly_theme['plot_bgcolor'],
                    height=500
                )
            
            if show_ma:
                # use_subplots is True for candlestick (has subplots), False for line chart
                use_subplots = (chart_type == "Candlestick")
                fig = add_moving_averages(fig, data, [20, 50], plotly_theme['colors'][1:], use_subplots=use_subplots)
            
            st.plotly_chart(fig, width='stretch')
            
            # Volume Analysis
            if show_volume and chart_type != "Candlestick":
                vol_fig = create_volume_analysis_chart(data, theme=plotly_theme)
                st.plotly_chart(vol_fig, width='stretch')
        
        # ============== TAB 2: TECHNICAL ANALYSIS ==============
        with tab2:
            st.markdown(render_section_header("Technical Indicators", "📊"), unsafe_allow_html=True)
            
            # Indicator Signals Summary
            signals_html = '<div style="margin: 1rem 0;">'
            if indicator_summary['rsi']['value']:
                signals_html += render_indicator_signal("RSI", indicator_summary['rsi']['signal'])
            if indicator_summary['macd']['signal']:
                signals_html += render_indicator_signal("MACD", indicator_summary['macd']['signal'])
            if indicator_summary['bollinger']['signal']:
                signals_html += render_indicator_signal("Bollinger", indicator_summary['bollinger']['signal'])
            signals_html += '</div>'
            st.markdown(signals_html, unsafe_allow_html=True)
            
            # Indicator Values
            ind_col1, ind_col2, ind_col3 = st.columns(3)
            with ind_col1:
                if indicator_summary['rsi']['value']:
                    st.metric("RSI (14)", f"{indicator_summary['rsi']['value']:.2f}")
            with ind_col2:
                if indicator_summary['sma_20']:
                    st.metric("SMA 20", f"${indicator_summary['sma_20']:.2f}")
            with ind_col3:
                if indicator_summary['atr']:
                    st.metric("ATR", f"${indicator_summary['atr']:.2f}")
            
            # Candlestick with Bollinger Bands
            st.markdown("### Price with Bollinger Bands")
            bb_fig = create_candlestick_chart(data, f"{symbol} with Bollinger Bands", False, plotly_theme)
            bb_fig = add_bollinger_bands(bb_fig, data)
            st.plotly_chart(bb_fig, width='stretch')
            
            # RSI Chart
            st.markdown("### RSI (Relative Strength Index)")
            rsi_fig = create_indicator_chart(data_with_indicators, 'RSI', theme=plotly_theme)
            st.plotly_chart(rsi_fig, width='stretch')
            
            # MACD Chart
            st.markdown("### MACD")
            macd_fig = create_indicator_chart(data_with_indicators, 'MACD', theme=plotly_theme)
            st.plotly_chart(macd_fig, width='stretch')
        
        # ============== TAB 3: AI FORECAST ==============
        with tab3:
            st.markdown(render_section_header("AI-Powered Price Forecast", "🔮"), unsafe_allow_html=True)
            
            # Forecast settings
            forecast_col1, forecast_col2 = st.columns(2)
            with forecast_col1:
                forecast_days = st.slider("Forecast Days", 7, 90, 30, help="Number of business days to forecast")
            with forecast_col2:
                confidence_level = st.slider("Confidence Level", 0.80, 0.99, 0.95, 0.01)
            
            if st.button("Generate AI Forecast", type="primary"):
                with st.spinner("Training LSTM model... This may take a minute."):
                    try:
                        # Use the improved training pipeline with proper data splitting
                        model, scaler, metrics, historical_errors = lstm.train_and_evaluate(data, epochs=15)

                        # Make historical predictions for visualization
                        predictions = lstm.make_predictions(model, data, scaler)

                        # Generate future forecast
                        future_dates, future_predictions = lstm.forecast_future(model, data, scaler, forecast_days)

                        st.session_state.model_trained = True
                        st.session_state.predictions = predictions
                        st.session_state.future_dates = future_dates
                        st.session_state.future_predictions = future_predictions
                        st.session_state.metrics = metrics
                        st.session_state.scaler = scaler
                        st.session_state.historical_errors = historical_errors

                        st.success("Model trained successfully!")
                        logger.info(f"Model trained for {symbol} with R²={metrics['r2']:.4f}")
                    except Exception as e:
                        st.error(f"Error training model: {str(e)}")
                        logger.error(f"Model training failed: {str(e)}")
            
            # Display results if model is trained
            if st.session_state.get('model_trained') and st.session_state.get('predictions') is not None:
                # Model Performance Metrics
                st.markdown("### Model Performance")
                metrics = st.session_state.metrics
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                with perf_col1:
                    st.metric("MAE", f"${metrics['mae']:.2f}")
                with perf_col2:
                    st.metric("RMSE", f"${metrics['rmse']:.2f}")
                with perf_col3:
                    st.metric("R² Score", f"{metrics['r2']:.4f}")
                with perf_col4:
                    st.metric("MAPE", f"{metrics['mape']:.2f}%")
                
                # Prediction Chart
                st.markdown("### Price Prediction with Future Forecast")
                pred_fig = create_prediction_chart(
                    data,
                    st.session_state.predictions,
                    st.session_state.future_dates,
                    st.session_state.future_predictions,
                    confidence_interval=1-confidence_level,
                    title=f"{symbol} Price Forecast",
                    theme=plotly_theme
                )
                st.plotly_chart(pred_fig, width='stretch')
                
                # Future Predictions Table
                st.markdown("### Forecast Details")
                forecast_df = lstm.create_prediction_dataframe(
                    st.session_state.future_dates,
                    st.session_state.future_predictions,
                    confidence_level,
                    historical_errors=st.session_state.get('historical_errors')
                )
                
                # Show first and last few predictions
                st.dataframe(
                    forecast_df.style.format({
                        'Predicted': '${:,.2f}',
                        'Upper': '${:,.2f}',
                        'Lower': '${:,.2f}'
                    }),
                    width='stretch'
                )
        
        # ============== TAB 4: COMPARE STOCKS ==============
        with tab4:
            st.markdown(render_section_header("Stock Comparison", "📉"), unsafe_allow_html=True)
            
            # Multi-stock selector
            all_symbols = list(STOCK_DATABASE.keys())
            compare_symbols = st.multiselect(
                "Select stocks to compare (max 5)",
                options=[s for s in all_symbols if s != symbol],
                max_selections=4,
                default=[]
            )
            
            # Add current stock
            compare_list = [symbol] + compare_symbols
            
            if len(compare_list) >= 2:
                # Load data for comparison
                with st.spinner("Loading comparison data..."):
                    comparison_data = load_multiple_stocks(compare_list, START_DATE, TODAY)
                
                # Comparison options
                comp_col1, comp_col2 = st.columns(2)
                with comp_col1:
                    normalize = st.checkbox("Normalize prices (% change)", value=True)
                with comp_col2:
                    time_period = st.selectbox("Time Period", ["1M", "3M", "6M", "1Y", "All"], index=3)
                
                # Filter by time period
                period_days = {"1M": 22, "3M": 66, "6M": 132, "1Y": 252, "All": None}
                days = period_days[time_period]
                
                if days:
                    for sym in comparison_data:
                        comparison_data[sym] = comparison_data[sym].tail(days)
                
                # Create comparison chart
                comp_fig = create_comparison_chart(
                    comparison_data,
                    normalize=normalize,
                    title="Stock Performance Comparison",
                    theme=plotly_theme
                )
                st.plotly_chart(comp_fig, width='stretch')
                
                # Comparison metrics table
                st.markdown("### Performance Metrics")
                
                metrics_data = []
                for sym, df in comparison_data.items():
                    info = get_stock_info(sym)
                    price_info = get_current_price_info(df)
                    rets = calculate_returns(df)
                    
                    metrics_data.append({
                        "Symbol": sym,
                        "Name": info['name'][:30] if info else sym,
                        "Price": f"${price_info['current_price']:.2f}" if price_info else "N/A",
                        "1W": f"{rets.get('1W', 0):+.1f}%" if rets else "N/A",
                        "1M": f"{rets.get('1M', 0):+.1f}%" if rets else "N/A",
                        "1Y": f"{rets.get('1Y', 0):+.1f}%" if rets else "N/A",
                    })
                
                st.dataframe(pd.DataFrame(metrics_data), width='stretch')
            else:
                st.info("Select at least one more stock to compare")
        
        # ============== TAB 5: EXPORT DATA ==============
        with tab5:
            st.markdown(render_section_header("Export Data", "📥"), unsafe_allow_html=True)
            
            export_col1, export_col2 = st.columns(2)
            
            with export_col1:
                st.markdown("### Historical Data")
                st.markdown(f"Export {len(data)} rows of historical OHLCV data")
                
                csv_data = data.to_csv(index=False)
                st.download_button(
                    label="📥 Download Historical Data (CSV)",
                    data=csv_data,
                    file_name=f"{symbol}_historical_data.csv",
                    mime="text/csv"
                )
            
            with export_col2:
                st.markdown("### Data with Indicators")
                st.markdown("Export data with all calculated technical indicators")
                
                csv_indicators = data_with_indicators.to_csv(index=False)
                st.download_button(
                    label="📥 Download with Indicators (CSV)",
                    data=csv_indicators,
                    file_name=f"{symbol}_with_indicators.csv",
                    mime="text/csv"
                )
            
            # Forecast export
            if st.session_state.get('future_predictions') is not None:
                st.markdown("### Forecast Data")
                forecast_df = lstm.create_prediction_dataframe(
                    st.session_state.future_dates,
                    st.session_state.future_predictions,
                    0.95,
                    historical_errors=st.session_state.get('historical_errors')
                )
                csv_forecast = forecast_df.to_csv(index=False)
                st.download_button(
                    label="Download Forecast (CSV)",
                    data=csv_forecast,
                    file_name=f"{symbol}_forecast.csv",
                    mime="text/csv"
                )

else:
    # No stock selected - show welcome message
    st.markdown("""
    <div class="glass-card" style="text-align: center; padding: 3rem;">
        <h2>Welcome to the S&P 500 Stock Predictor! 🚀</h2>
        <p style="font-size: 1.1rem; color: #888;">
            Select a stock from the sidebar to get started with advanced analysis and AI-powered forecasting.
        </p>
        <div style="margin-top: 2rem;">
            <h3>Features:</h3>
            <ul style="text-align: left; display: inline-block;">
                <li>📈 Interactive Candlestick Charts</li>
                <li>📊 Technical Indicators (RSI, MACD, Bollinger Bands)</li>
                <li>🔮 AI-Powered Price Forecasting</li>
                <li>📉 Multi-Stock Comparison</li>
                <li>📥 Data Export to CSV</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show popular stocks
    st.markdown(render_section_header("Popular Stocks", "🔥"), unsafe_allow_html=True)
    
    popular = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA"]
    pop_cols = st.columns(6)
    
    for i, sym in enumerate(popular):
        with pop_cols[i]:
            info = get_stock_info(sym)
            if st.button(f"**{sym}**\n{info['name'][:15]}..." if info else sym, key=f"pop_{sym}"):
                st.session_state.selected_symbol = sym
                st.rerun()


# Footer with disclaimer
st.markdown("""
<div class="footer">
    <p>Built with Streamlit, TensorFlow, and Plotly</p>
    <p><strong>DISCLAIMER:</strong> This application is for educational purposes only.
    Predictions are based on historical data and machine learning models which may not
    accurately predict future prices. This is NOT financial advice. Always consult a
    qualified financial advisor before making investment decisions.</p>
    <p>Data provided by Yahoo Finance</p>
</div>
""", unsafe_allow_html=True)
