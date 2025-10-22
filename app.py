import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
    .section-header {
        font-size: 1.5rem;
        color: #2E86AB;
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
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
            self.data = self.data.ffill().bfill()
            
            # Create derived features
            self.create_technical_indicators()
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
    
    def create_technical_indicators(self):
        """Create technical indicators and derived features"""
        # Moving averages
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        
        # Price changes and returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Price_Change'] = self.data['Close'] - self.data['Open']
        self.data['Cumulative_Return'] = (1 + self.data['Daily_Return']).cumprod()
        
        # Volatility
        self.data['Volatility_20'] = self.data['Daily_Return'].rolling(window=20).std()
        
        # RSI
        self.data['RSI'] = self.calculate_rsi()
        
        # MACD
        self.data['MACD'], self.data['MACD_Signal'] = self.calculate_macd()
        
        # Bollinger Bands
        self.data['BB_Upper'], self.data['BB_Lower'], self.data['BB_Middle'] = self.calculate_bollinger_bands()
        
    def calculate_rsi(self, period=14):
        """Calculate Relative Strength Index"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self):
        """Calculate MACD indicator"""
        exp1 = self.data['Close'].ewm(span=12).mean()
        exp2 = self.data['Close'].ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def calculate_bollinger_bands(self, period=20):
        """Calculate Bollinger Bands"""
        sma = self.data['Close'].rolling(window=period).mean()
        std = self.data['Close'].rolling(window=period).std()
        upper_band = sma + (std * 2)
        lower_band = sma - (std * 2)
        return upper_band, lower_band, sma
    
    def get_descriptive_stats(self):
        """Calculate descriptive statistics"""
        stats = {
            'Mean Price': self.data['Close'].mean(),
            'Median Price': self.data['Close'].median(),
            'Std Deviation': self.data['Close'].std(),
            'Min Price': self.data['Close'].min(),
            'Max Price': self.data['Close'].max(),
            'Total Return': (self.data['Close'].iloc[-1] - self.data['Close'].iloc[0]) / self.data['Close'].iloc[0] * 100,
            'Average Volume': self.data['Volume'].mean(),
            'Volatility (Annual)': self.data['Daily_Return'].std() * np.sqrt(252) * 100,
            'Sharpe Ratio': self.data['Daily_Return'].mean() / self.data['Daily_Return'].std() * np.sqrt(252) if self.data['Daily_Return'].std() != 0 else 0
        }
        return stats
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix for numerical features"""
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD']
        available_cols = [col for col in numerical_cols if col in self.data.columns]
        corr_data = self.data[available_cols].corr()
        return corr_data
    
    def create_time_series_plot(self):
        """Create interactive time series plot with range selector"""
        fig = go.Figure()
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Close'],
            name='Close Price',
            line=dict(color='blue', width=2)
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data['MA_20'],
            name='20-Day MA',
            line=dict(color='orange', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data['MA_50'],
            name='50-Day MA',
            line=dict(color='red', width=1)
        ))
        
        fig.add_trace(go.Scatter(
            x=self.data.index, 
            y=self.data['MA_200'],
            name='200-Day MA',
            line=dict(color='purple', width=1)
        ))
        
        fig.update_layout(
            title=f'{self.ticker} Stock Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_candlestick_chart(self):
        """Create detailed candlestick chart"""
        fig = go.Figure(data=[go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='Price'
        )])
        
        fig.update_layout(
            title=f'{self.ticker} Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=True,
            height=500
        )
        
        return fig
    
    def create_correlation_heatmap(self):
        """Create interactive correlation heatmap"""
        corr_matrix = self.get_correlation_matrix()
        
        fig = px.imshow(
            corr_matrix,
            title=f'{self.ticker} Feature Correlation Matrix',
            color_continuous_scale='RdBu_r',
            aspect='auto',
            text_auto=True
        )
        
        fig.update_layout(height=500)
        return fig
    
    def create_volume_chart(self):
        """Create volume analysis chart"""
        fig = go.Figure()
        
        # Color bars based on price change
        colors = ['red' if x < 0 else 'green' for x in self.data['Price_Change']]
        
        fig.add_trace(go.Bar(
            x=self.data.index,
            y=self.data['Volume'],
            name='Volume',
            marker_color=colors,
            opacity=0.7
        ))
        
        fig.update_layout(
            title=f'{self.ticker} Trading Volume (Red = Down Day, Green = Up Day)',
            xaxis_title='Date',
            yaxis_title='Volume',
            height=400
        )
        
        return fig
    
    def create_technical_indicators_plot(self):
        """Create subplot with multiple technical indicators"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('RSI (Relative Strength Index)', 'MACD', 'Bollinger Bands'),
            vertical_spacing=0.08,
            row_heights=[0.3, 0.3, 0.4]
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['RSI'], name='RSI'),
            row=1, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=1, col=1)
        
        # MACD
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['MACD'], name='MACD'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['MACD_Signal'], name='Signal'),
            row=2, col=1
        )
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Close'], name='Price'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['BB_Upper'], name='Upper Band', line=dict(dash='dash')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['BB_Lower'], name='Lower Band', line=dict(dash='dash')),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['BB_Middle'], name='Middle Band', line=dict(dash='dot')),
            row=3, col=1
        )
        fig.update_yaxes(title_text="Price", row=3, col=1)
        
        fig.update_layout(height=800, title_text=f"{self.ticker} Technical Indicators", showlegend=True)
        return fig
    
    def create_returns_distribution(self):
        """Create histogram of daily returns"""
        returns_data = self.data['Daily_Return'].dropna()
        
        fig = px.histogram(
            x=returns_data,
            title=f'{self.ticker} Daily Returns Distribution',
            nbins=50,
            color_discrete_sequence=['lightblue'],
            opacity=0.7
        )
        
        fig.update_layout(
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            height=400
        )
        
        # Add mean line
        mean_return = returns_data.mean()
        fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_return:.4f}")
        
        return fig
    
    def create_volatility_chart(self):
        """Create volatility analysis chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.data.index,
            y=self.data['Volatility_20'],
            name='20-Day Volatility',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{self.ticker} 20-Day Rolling Volatility',
            xaxis_title='Date',
            yaxis_title='Volatility',
            height=400
        )
        
        return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Advanced Stock Market Analysis Dashboard</h1>', 
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
        "Meta Platforms": "META",
        "NVIDIA": "NVDA",
        "JPMorgan Chase": "JPM"
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
    
    # Analysis focus
    analysis_focus = st.sidebar.multiselect(
        "Analysis Focus:",
        ["Technical Indicators", "Price Analysis", "Volume Analysis", "Statistical Analysis"],
        default=["Technical Indicators", "Price Analysis"]
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
        st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)
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
                label="Annual Volatility",
                value=f"{stats.get('Volatility (Annual)', 0):.2f}%"
            )
        
        with col4:
            st.metric(
                label="Sharpe Ratio",
                value=f"{stats.get('Sharpe Ratio', 0):.2f}"
            )
        
        # Additional metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            st.metric("Average Volume", f"{stats.get('Average Volume', 0):,.0f}")
        with col6:
            st.metric("Data Points", len(data))
        with col7:
            st.metric("Start Date", data.index[0].strftime('%Y-%m-%d'))
        with col8:
            st.metric("End Date", data.index[-1].strftime('%Y-%m-%d'))
        
        # Tabs for different analyses
        tab_names = []
        if "Price Analysis" in analysis_focus:
            tab_names.append("📈 Price Analysis")
        if "Technical Indicators" in analysis_focus:
            tab_names.append("🔧 Technical Analysis")
        if "Volume Analysis" in analysis_focus:
            tab_names.append("📊 Volume & Statistics")
        if "Statistical Analysis" in analysis_focus:
            tab_names.append("📋 Advanced Stats")
        tab_names.append("ℹ️ Documentation")
        
        tabs = st.tabs(tab_names)
        
        current_tab = 0
        
        if "Price Analysis" in analysis_focus:
            with tabs[current_tab]:
                st.markdown('<div class="section-header">Price Analysis</div>', unsafe_allow_html=True)
                
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
                    recent_data = data.tail(10)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
                    recent_data['Change'] = recent_data['Close'] - recent_data['Open']
                    recent_data['Change %'] = (recent_data['Change'] / recent_data['Open']) * 100
                    
                    st.dataframe(recent_data.style.format({
                        'Open': '${:.2f}', 'High': '${:.2f}', 'Low': '${:.2f}', 
                        'Close': '${:.2f}', 'Change': '${:.2f}', 'Change %': '{:.2f}%',
                        'Volume': '{:,.0f}'
                    }))
                
                # Cumulative returns
                st.subheader("Cumulative Returns")
                cum_fig = go.Figure()
                cum_fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Cumulative_Return'],
                    name='Cumulative Return',
                    line=dict(color='green', width=2)
                ))
                cum_fig.update_layout(
                    title="Cumulative Returns Over Time",
                    xaxis_title="Date",
                    yaxis_title="Cumulative Return",
                    height=400
                )
                st.plotly_chart(cum_fig, use_container_width=True)
                
            current_tab += 1
        
        if "Technical Indicators" in analysis_focus:
            with tabs[current_tab]:
                st.markdown('<div class="section-header">Technical Analysis</div>', unsafe_allow_html=True)
                
                # Technical indicators subplot
                fig_tech = analyzer.create_technical_indicators_plot()
                st.plotly_chart(fig_tech, use_container_width=True)
                
                # Technical indicators explanation
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📈 RSI Interpretation")
                    st.info("""
                    - **RSI > 70**: Potentially overbought (Consider selling)
                    - **RSI < 30**: Potentially oversold (Consider buying)
                    - **30-70**: Normal range
                    - **Divergences**: Can signal trend reversals
                    """)
                    
                    st.subheader("📊 Bollinger Bands")
                    st.info("""
                    - **Price touches upper band**: Potentially overbought
                    - **Price touches lower band**: Potentially oversold
                    - **Band squeeze**: Low volatility, potential breakout coming
                    - **Band expansion**: High volatility, strong trend
                    """)
                
                with col2:
                    st.subheader("⚡ MACD Signals")
                    st.info("""
                    - **MACD > Signal**: Bullish momentum
                    - **MACD < Signal**: Bearish momentum
                    - **Zero line crossover**: Trend change signal
                    - **Divergence**: Potential reversal signal
                    """)
                    
                    st.subheader("📏 Moving Averages")
                    st.info("""
                    - **Golden Cross**: 50-day MA crosses above 200-day MA (Bullish)
                    - **Death Cross**: 50-day MA crosses below 200-day MA (Bearish)
                    - **Price above MA**: Uptrend
                    - **Price below MA**: Downtrend
                    """)
                
            current_tab += 1
        
        if "Volume Analysis" in analysis_focus:
            with tabs[current_tab]:
                st.markdown('<div class="section-header">Volume & Statistical Analysis</div>', unsafe_allow_html=True)
                
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
                    
                    # Volatility chart
                    fig_vol = analyzer.create_volatility_chart()
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                # Scatter plot: Volume vs Price Change
                st.subheader("Volume vs Price Change Analysis")
                scatter_data = data.reset_index()
                fig_scatter = px.scatter(
                    scatter_data,
                    x='Volume',
                    y='Price_Change',
                    title='Volume vs Daily Price Change',
                    trendline='lowess',
                    color='Price_Change',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
            current_tab += 1
        
        if "Statistical Analysis" in analysis_focus:
            with tabs[current_tab]:
                st.markdown('<div class="section-header">Advanced Statistical Analysis</div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Descriptive Statistics")
                    if stats:
                        stats_df = pd.DataFrame.from_dict(stats, orient='index', columns=['Value'])
                        stats_df.index.name = 'Metric'
                        st.dataframe(stats_df.style.format({'Value': '{:,.4f}'}))
                
                with col2:
                    st.subheader("Correlation Matrix")
                    corr_matrix = analyzer.get_correlation_matrix()
                    st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None).format('{:.3f}'))
                
                # Additional statistical insights
                st.subheader("📈 Performance Insights")
                
                col3, col4, col5, col6 = st.columns(4)
                
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
                
                with col6:
                    avg_daily_return = data['Daily_Return'].mean() * 100
                    st.metric("Average Daily Return", f"{avg_daily_return:.2f}%")
                
                # Risk Analysis
                st.subheader("⚡ Risk Analysis")
                col7, col8, col9, col10 = st.columns(4)
                
                with col7:
                    # Value at Risk (95% confidence)
                    var_95 = data['Daily_Return'].quantile(0.05) * 100
                    st.metric("1-Day VaR (95%)", f"{var_95:.2f}%")
                
                with col8:
                    # Expected Shortfall
                    es_95 = data['Daily_Return'][data['Daily_Return'] <= data['Daily_Return'].quantile(0.05)].mean() * 100
                    st.metric("Expected Shortfall", f"{es_95:.2f}%")
                
                with col9:
                    # Skewness
                    skewness = data['Daily_Return'].skew()
                    st.metric("Return Skewness", f"{skewness:.3f}")
                
                with col10:
                    # Kurtosis
                    kurtosis = data['Daily_Return'].kurtosis()
                    st.metric("Return Kurtosis", f"{kurtosis:.3f}")
                
            current_tab += 1
        
        # Documentation tab
        with tabs[current_tab]:
            st.markdown('<div class="section-header">Documentation & How to Use</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 About This Dashboard")
                st.markdown("""
                This advanced stock market analysis dashboard provides comprehensive technical and fundamental analysis including:
                
                ### 📈 Core Features
                - **Real-time Price Data**: Live stock data from Yahoo Finance
                - **Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages
                - **Advanced Charting**: Interactive candlestick, line, and volume charts
                - **Statistical Analysis**: Descriptive statistics, correlations, risk metrics
                - **Risk Management**: Volatility analysis, Value at Risk, performance metrics
                
                ### 🔧 Technical Indicators Explained
                - **RSI (Relative Strength Index)**: Momentum oscillator measuring speed and change of price movements
                - **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
                - **Bollinger Bands**: Volatility bands placed above and below a moving average
                - **Moving Averages**: Smooth out price data to identify trends
                
                ### 📊 Statistical Metrics
                - **Sharpe Ratio**: Risk-adjusted return measure
                - **Value at Risk (VaR)**: Maximum potential loss at a given confidence level
                - **Volatility**: Standard deviation of returns, measures risk
                - **Skewness & Kurtosis**: Distribution characteristics of returns
                """)
            
            with col2:
                st.subheader("🚀 How to Use")
                st.info("""
                1. **Select Stock**: Choose from popular stocks
                2. **Set Time Period**: 1-5 years of historical data
                3. **Choose Analysis**: Select focus areas
                4. **Navigate Tabs**: Explore different analyses
                5. **Interact**: Hover over charts for details
                6. **Zoom**: Use range selectors on charts
                """)
                
                st.subheader("📚 Data Sources")
                st.success("""
                - **Yahoo Finance**: Real-time stock data
                - **Automatic Updates**: Daily market data
                - **Technical Calculations**: Real-time indicator computation
                - **Historical Analysis**: Multi-year backtesting
                """)
                
                st.subheader("⚠️ Disclaimer")
                st.warning("""
                This tool is for educational and analytical purposes only. 
                Not financial advice. Always conduct your own research 
                and consult with qualified financial advisors before 
                making investment decisions.
                """)
    
    except Exception as e:
        st.error(f"Error in application: {str(e)}")
        st.info("Please try refreshing the page or selecting a different stock/time period.")

if __name__ == "__main__":
    main()
