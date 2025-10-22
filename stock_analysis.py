import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
            print(f"Error loading data: {e}")
    
    def create_technical_indicators(self):
        """Create technical indicators and derived features"""
        # Moving averages
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['MA_200'] = self.data['Close'].rolling(window=200).mean()
        
        # Price changes and returns
        self.data['Daily_Return'] = self.data['Close'].pct_change()
        self.data['Price_Change'] = self.data['Close'] - self.data['Open']
        
        # Volatility
        self.data['Volatility'] = self.data['Daily_Return'].rolling(window=20).std()
        
        # RSI
        self.data['RSI'] = self.calculate_rsi()
        
        # MACD
        self.data['MACD'], self.data['MACD_Signal'] = self.calculate_macd()
        
        # Bollinger Bands
        self.data['BB_Upper'], self.data['BB_Lower'] = self.calculate_bollinger_bands()
        
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
        return upper_band, lower_band
    
    def get_descriptive_stats(self):
        """Calculate descriptive statistics"""
        stats = {
            'Mean Price': self.data['Close'].mean(),
            'Median Price': self.data['Close'].median(),
            'Std Deviation': self.data['Close'].std(),
            'Min Price': self.data['Close'].min(),
            'Max Price': self.data['Close'].max(),
            'Total Return': (self.data['Close'][-1] - self.data['Close'][0]) / self.data['Close'][0] * 100,
            'Average Volume': self.data['Volume'].mean(),
            'Volatility (Annual)': self.data['Daily_Return'].std() * np.sqrt(252) * 100
        }
        return stats
    
    def get_correlation_matrix(self):
        """Calculate correlation matrix for numerical features"""
        numerical_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD']
        corr_data = self.data[numerical_cols].corr()
        return corr_data
    
    def create_time_series_plot(self):
        """Create interactive time series plot with range selector"""
        fig = go.Figure()
        
        # Add candlestick trace
        fig.add_trace(go.Candlestick(
            x=self.data.index,
            open=self.data['Open'],
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            name='Price'
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
        
        fig.update_layout(
            title=f'{self.ticker} Stock Price with Moving Averages',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=False,
            height=500
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
            aspect='auto'
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
            title=f'{self.ticker} Trading Volume',
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
        
        # MACD
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['MACD'], name='MACD'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['MACD_Signal'], name='Signal'),
            row=2, col=1
        )
        
        # Bollinger Bands
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['Close'], name='Price'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['BB_Upper'], name='Upper Band'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=self.data.index, y=self.data['BB_Lower'], name='Lower Band'),
            row=3, col=1
        )
        
        fig.update_layout(height=800, title_text=f"{self.ticker} Technical Indicators")
        return fig
    
    def create_returns_distribution(self):
        """Create histogram of daily returns"""
        fig = px.histogram(
            self.data, 
            x='Daily_Return',
            title=f'{self.ticker} Daily Returns Distribution',
            nbins=50,
            color_discrete_sequence=['lightblue']
        )
        
        fig.update_layout(
            xaxis_title='Daily Return',
            yaxis_title='Frequency',
            height=400
        )
        
        # Add mean line
        mean_return = self.data['Daily_Return'].mean()
        fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                     annotation_text=f"Mean: {mean_return:.4f}")
        
        return fig