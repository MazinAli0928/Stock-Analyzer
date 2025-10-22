# app.py - Working Stock Analysis Dashboard
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page config FIRST
st.set_page_config(
    page_title="Stock Market Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add a simple title to test if app loads
st.title("📈 Stock Market Analysis Dashboard")
st.markdown("Select a stock and time period from the sidebar to begin analysis.")

# Sidebar controls
st.sidebar.title("Controls")
ticker = st.sidebar.selectbox(
    "Select Stock:",
    ["AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NFLX", "META"]
)

period = st.sidebar.selectbox(
    "Time Period:",
    ["1y", "2y", "3y", "5y"],
    index=2
)

# Simple data loading function
@st.cache_data
def load_stock_data(ticker, period):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

# Load data
data = load_stock_data(ticker, period)

if data.empty:
    st.warning("No data loaded. Please check your internet connection and try again.")
else:
    # Calculate basic metrics
    current_price = data['Close'].iloc[-1]
    price_change = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]) * 100
    volume_avg = data['Volume'].mean()
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        st.metric("Total Return", f"{price_change:.2f}%")
    with col3:
        st.metric("Avg Volume", f"{volume_avg:,.0f}")
    with col4:
        st.metric("Data Points", len(data))
    
    # Tab 1: Basic Price Chart
    st.subheader(f"{ticker} Price Chart")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title=f"{ticker} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Tab 2: Candlestick Chart
    st.subheader(f"{ticker} Candlestick Chart")
    candlestick = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    )])
    candlestick.update_layout(
        title=f"{ticker} Candlestick Chart",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        xaxis_rangeslider_visible=True
    )
    st.plotly_chart(candlestick, use_container_width=True)
    
    # Tab 3: Volume Chart
    st.subheader("Trading Volume")
    colors = ['red' if data['Close'].iloc[i] < data['Open'].iloc[i] else 'green' 
              for i in range(len(data))]
    
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        marker_color=colors,
        opacity=0.7
    ))
    volume_fig.update_layout(
        title="Trading Volume (Red = Down Day, Green = Up Day)",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=400
    )
    st.plotly_chart(volume_fig, use_container_width=True)
    
    # Tab 4: Statistical Analysis
    st.subheader("Statistical Analysis")
    
    # Descriptive Statistics
    st.write("**Descriptive Statistics:**")
    stats_data = {
        'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '25% Quantile', '75% Quantile'],
        'Value': [
            data['Close'].mean(),
            data['Close'].median(),
            data['Close'].std(),
            data['Close'].min(),
            data['Close'].max(),
            data['Close'].quantile(0.25),
            data['Close'].quantile(0.75)
        ]
    }
    stats_df = pd.DataFrame(stats_data)
    stats_df['Value'] = stats_df['Value'].round(2)
    st.dataframe(stats_df)
    
    # Correlation Matrix
    st.write("**Correlation Matrix:**")
    numeric_data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    corr_matrix = numeric_data.corr()
    
    fig_corr = px.imshow(
        corr_matrix,
        title="Feature Correlation Matrix",
        color_continuous_scale='RdBu_r',
        aspect='auto'
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Tab 5: Returns Analysis
    st.subheader("Returns Analysis")
    
    # Calculate daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    returns_data = data['Daily_Return'].dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Returns distribution
        st.write("**Daily Returns Distribution:**")
        hist_fig = px.histogram(
            x=returns_data,
            nbins=50,
            title="Distribution of Daily Returns",
            color_discrete_sequence=['lightblue']
        )
        hist_fig.add_vline(x=returns_data.mean(), line_dash="dash", line_color="red")
        st.plotly_chart(hist_fig, use_container_width=True)
    
    with col2:
        # Cumulative returns
        st.write("**Cumulative Returns:**")
        data['Cumulative_Return'] = (1 + data['Daily_Return']).cumprod()
        cum_fig = px.line(
            x=data.index,
            y=data['Cumulative_Return'],
            title="Cumulative Returns Over Time"
        )
        st.plotly_chart(cum_fig, use_container_width=True)
    
    # Additional Info
    st.subheader("Dataset Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Data Range:** {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        st.write(f"**Total Trading Days:** {len(data)}")
        st.write(f"**Missing Values:** {data.isnull().sum().sum()}")
    
    with col2:
        positive_days = (data['Daily_Return'] > 0).sum()
        total_days = len(returns_data)
        st.write(f"**Positive Days:** {positive_days}/{total_days} ({positive_days/total_days*100:.1f}%)")
        st.write(f"**Largest Gain:** {returns_data.max()*100:.2f}%")
        st.write(f"**Largest Loss:** {returns_data.min()*100:.2f}%")

# Add footer
st.markdown("---")
st.markdown("Built with Streamlit • Data from Yahoo Finance")

