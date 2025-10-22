import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from stock_analysis import StockAnalyzer

# Page configuration
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Stock Market Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("🔧 Controls & Filters")
    
    # Stock selection
    ticker_options = {
        "Apple Inc.": "AAPL",
        "Tesla Inc.": "TSLA", 
        "Google (Alphabet)": "GOOGL",
        "Microsoft": "MSFT",
        "Amazon": "AMZN",
        "Netflix": "NFLX",
        "Meta Platforms": "META"
    }
    
    selected_stock_name = st.sidebar.selectbox(
        "Select Stock:",
        list(ticker_options.keys())
    )
    ticker = ticker_options[selected_stock_name]
    
    # Time period selection
    period = st.sidebar.selectbox(
        "Time Period:",
        ["1y", "2y", "3y", "5y"],
        index=2
    )
    
    # Analysis type
    analysis_type = st.sidebar.radio(
        "Analysis Focus:",
        ["Technical Analysis", "Price Trends", "Volume Analysis", "Returns Analysis"]
    )
    
    # Initialize stock analyzer
    @st.cache_data
    def load_stock_data(ticker, period):
        return StockAnalyzer(ticker, period)
    
    try:
        analyzer = load_stock_data(ticker, period)
        data = analyzer.data
        
        if data.empty:
            st.error("No data available for the selected stock and period.")
            return
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics
        stats = analyzer.get_descriptive_stats()
        
        with col1:
            st.metric(
                label="Current Price",
                value=f"${data['Close'].iloc[-1]:.2f}",
                delta=f"{data['Daily_Return'].iloc[-1]:.2%}"
            )
        
        with col2:
            st.metric(
                label="Total Return",
                value=f"{stats['Total Return']:.2f}%"
            )
        
        with col3:
            st.metric(
                label="Volatility (Annual)",
                value=f"{stats['Volatility (Annual)']:.2f}%"
            )
        
        with col4:
            st.metric(
                label="Avg Daily Volume",
                value=f"{stats['Average Volume']:,.0f}"
            )
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Price Analysis", 
            "🔍 Technical Indicators", 
            "📈 Volume & Correlation",
            "📋 Statistics",
            "ℹ️ Stock Info"
        ])
        
        with tab1:
            st.header("Price Analysis")
            
            # Time series with moving averages
            fig_ts = analyzer.create_time_series_plot()
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Candlestick chart
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig_candle = analyzer.create_candlestick_chart()
                st.plotly_chart(fig_candle, use_container_width=True)
            
            with col2:
                st.subheader("Recent Performance")
                recent_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']]
                st.dataframe(recent_data.style.format({
                    'Open': '${:.2f}', 'High': '${:.2f}', 
                    'Low': '${:.2f}', 'Close': '${:.2f}',
                    'Volume': '{:,.0f}'
                }))
        
        with tab2:
            st.header("Technical Indicators")
            
            # Technical indicators subplot
            fig_tech = analyzer.create_technical_indicators_plot()
            st.plotly_chart(fig_tech, use_container_width=True)
            
            # RSI explanation
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RSI Interpretation")
                st.info("""
                - **RSI > 70**: Potentially overbought
                - **RSI < 30**: Potentially oversold
                - **30-70**: Normal range
                """)
            
            with col2:
                st.subheader("MACD Signals")
                st.info("""
                - **MACD > Signal**: Bullish momentum
                - **MACD < Signal**: Bearish momentum
                - **Crossovers**: Potential trend changes
                """)
        
        with tab3:
            st.header("Volume & Correlation Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume analysis
                fig_volume = analyzer.create_volume_chart()
                st.plotly_chart(fig_volume, use_container_width=True)
                
                # Returns distribution
                fig_returns = analyzer.create_returns_distribution()
                st.plotly_chart(fig_returns, use_container_width=True)
            
            with col2:
                # Correlation heatmap
                fig_corr = analyzer.create_correlation_heatmap()
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Scatter plot: Volume vs Price Change
                fig_scatter = px.scatter(
                    data.reset_index(),
                    x='Volume',
                    y='Price_Change',
                    title='Volume vs Price Change',
                    trendline='lowess'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab4:
            st.header("Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Descriptive Statistics")
                stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
                stats_df.index.name = 'Metric'
                st.dataframe(stats_df.style.format({
                    'Value': '{:,.2f}'
                }))
            
            with col2:
                st.subheader("Correlation Matrix")
                corr_matrix = analyzer.get_correlation_matrix()
                st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm').format('{:.2f}'))
            
            # Additional statistical insights
            st.subheader("Statistical Insights")
            
            col3, col4, col5 = st.columns(3)
            
            with col3:
                positive_days = (data['Daily_Return'] > 0).sum()
                total_days = len(data['Daily_Return'].dropna())
                st.metric("Positive Trading Days", f"{positive_days}/{total_days} ({positive_days/total_days:.1%})")
            
            with col4:
                max_gain = data['Daily_Return'].max() * 100
                st.metric("Maximum Daily Gain", f"{max_gain:.2f}%")
            
            with col5:
                max_loss = data['Daily_Return'].min() * 100
                st.metric("Maximum Daily Loss", f"{max_loss:.2f}%")
        
        with tab5:
            st.header("Stock Information & Documentation")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("About This Dashboard")
                st.markdown("""
                This interactive dashboard provides comprehensive stock market analysis including:
                
                - **Price Trends**: Historical price data with moving averages
                - **Technical Indicators**: RSI, MACD, Bollinger Bands
                - **Volume Analysis**: Trading volume patterns
                - **Statistical Insights**: Returns distribution and correlations
                - **Risk Metrics**: Volatility and performance statistics
                
                ### Data Source
                - Real-time stock data from Yahoo Finance via `yfinance` library
                - Automatic technical indicator calculations
                - Interactive visualizations using Plotly
                """)
            
            with col2:
                st.subheader("How to Use")
                st.info("""
                1. Select stock from sidebar
                2. Choose time period
                3. Navigate through tabs
                4. Hover over charts for details
                5. Use range selectors to zoom
                """)
                
                st.subheader("Technical Notes")
                st.success("""
                - Data updates daily
                - All calculations in real-time
                - Mobile-responsive design
                - Open-source implementation
                """)
    
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please check your internet connection and try again.")

if __name__ == "__main__":
    main()