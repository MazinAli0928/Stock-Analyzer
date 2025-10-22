# app.py - Stock Market Analysis Dashboard (Python 3.13 Compatible)
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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
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

class StockAnalyzer:
    def __init__(self, ticker, period="3y"):
        self.ticker = ticker
        self.period = period
        self.data = None
        self.load_data()
        
    def load_data(self):
        """Load and clean stock data"""
        try:
            stock = yf.Ticker(self.ticker)
            self.data = stock.history(period=self.period)
            
            # Handle missing values
            if self.data.empty:
                st.error(f"No data available for {self.ticker}")
                return
                
            self.data = self.data.ffill().bfill()
            self.create_technical_indicators()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def create_technical_indicators(self):
        """Create technical indicators"""
        # Moving averages
        self.data['MA_20'] = self.data['Close'].rolling(window=20, min_periods=1).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50, min_periods=1).mean()
        
        # Price changes and returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Price_Change'] = self.data['Close'] - self.data['Open']
        
        # Volatility
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=20, min_periods=1).std()
        
        # RSI
        self.data['RSI'] = self.calculate_rsi()
    
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_descriptive_stats(self):
        """Calculate descriptive statistics"""
        if self.data.empty:
            return {}
            
        stats = {
            'Mean Price': self.data['Close'].mean(),
            'Median Price': self.data['Close'].median(),
            'Std Deviation': self.data['Close'].std(),
            'Min Price': self.data['Close'].min(),
            'Max Price': self.data['Close'].max(),
            'Total Return': (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100,
            'Average Volume': self.data['Volume'].mean(),
            'Volatility (Annual)': self.data['Daily_Return'].std() * np.sqrt(252) * 100
        }
        return stats

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
    
    # Initialize stock analyzer with caching
    @st.cache_data
    def load_stock_analyzer(_ticker, _period):
        return StockAnalyzer(_ticker, _period)
    
    try:
        analyzer = load_stock_analyzer(ticker, period)
        data = analyzer.data
        
        if data.empty:
            st.error("No data available for the selected stock and period.")
            return
        
        # Main dashboard layout
        col1, col2, col3, col4 = st.columns(4)
        
        # Key metrics
        stats = analyzer.get_descriptive_stats()
        
        with col1:
            current_return = data['Daily_Return'].iloc[-1] if not pd.isna(data['Daily_Return'].iloc[-1]) else 0
            st.metric(
                label="Current Price",
                value=f"${data['Close'].iloc[-1]:.2f}",
                delta=f"{current_return:.2%}"
            )
        
        with col2:
            st.metric(
                label="Total Return",
                value=f"{stats.get('Total Return', 0):.2f}%"
            )
        
        with col3:
            st.metric(
                label="Volatility (Annual)",
                value=f"{stats.get('Volatility (Annual)', 0):.2f}%"
            )
        
        with col4:
            st.metric(
                label="Avg Daily Volume",
                value=f"{stats.get('Average Volume', 0):,.0f}"
            )
        
        # Tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Price Analysis", 
            "🔍 Technical Indicators", 
            "📈 Volume & Statistics",
            "ℹ️ About"
        ])
        
        with tab1:
            st.header("Price Analysis")
            
            # Price chart with moving averages
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Close'],
                name='Close Price',
                line=dict(color='blue', width=2)
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MA_20'],
                name='20-Day MA',
                line=dict(color='orange', width=1)
            ))
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['MA_50'],
                name='50-Day MA',
                line=dict(color='red', width=1)
            ))
            fig.update_layout(
                title=f'{ticker} Stock Price with Moving Averages',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Candlestick chart
            st.subheader("Candlestick Chart")
            candlestick = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            )])
            candlestick.update_layout(
                title=f'{ticker} Candlestick Chart',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                height=500,
                xaxis_rangeslider_visible=True
            )
            st.plotly_chart(candlestick, use_container_width=True)
        
        with tab2:
            st.header("Technical Indicators")
            
            # RSI
            st.subheader("RSI (Relative Strength Index)")
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['RSI'],
                name='RSI',
                line=dict(color='purple', width=2)
            ))
            rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
            rsi_fig.update_layout(
                title="RSI - Overbought (>70) vs Oversold (<30)",
                xaxis_title='Date',
                yaxis_title='RSI',
                height=400
            )
            st.plotly_chart(rsi_fig, use_container_width=True)
            
            # Volatility
            st.subheader("Volatility")
            vol_fig = go.Figure()
            vol_fig.add_trace(go.Scatter(
                x=data.index,
                y=data['Volatility'],
                name='Volatility',
                line=dict(color='brown', width=2)
            ))
            vol_fig.update_layout(
                title="20-Day Rolling Volatility",
                xaxis_title='Date',
                yaxis_title='Volatility',
                height=400
            )
            st.plotly_chart(vol_fig, use_container_width=True)
        
        with tab3:
            st.header("Volume & Statistical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Volume chart
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
            
            with col2:
                # Returns distribution
                st.subheader("Daily Returns Distribution")
                returns_data = data['Daily_Return'].dropna()
                
                hist_fig = px.histogram(
                    x=returns_data,
                    nbins=50,
                    title="Distribution of Daily Returns",
                    color_discrete_sequence=['lightblue']
                )
                hist_fig.add_vline(x=returns_data.mean(), line_dash="dash", line_color="red")
                hist_fig.update_layout(height=400)
                st.plotly_chart(hist_fig, use_container_width=True)
            
            # Statistical summary
            st.subheader("Descriptive Statistics")
            if stats:
                stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
                stats_df.index.name = 'Metric'
                st.dataframe(stats_df.style.format({'Value': '{:,.2f}'}))
        
        with tab4:
            st.header("About This Dashboard")
            
            st.markdown("""
            ### Stock Market Analysis Dashboard
            
            This interactive dashboard provides comprehensive stock market analysis including:
            
            - **Price Trends**: Historical price data with moving averages
            - **Technical Indicators**: RSI and volatility analysis
            - **Volume Analysis**: Trading volume patterns
            - **Statistical Insights**: Returns distribution and performance metrics
            
            ### Features
            - Real-time stock data from Yahoo Finance
            - Interactive charts with Plotly
            - Technical analysis indicators
            - Responsive design for all devices
            
            ### Data Source
            - Stock data provided by Yahoo Finance
            - Automatic updates during market hours
            - Historical data analysis
            
            ### How to Use
            1. Select a stock from the sidebar
            2. Choose your analysis time period
            3. Navigate through different analysis tabs
            4. Hover over charts for detailed information
            """)
    
    except Exception as e:
        st.error(f"Error in application: {str(e)}")
        st.info("Please try refreshing the page or selecting a different stock.")

if __name__ == "__main__":
    main()
