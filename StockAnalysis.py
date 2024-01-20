# path/filename: stock_analysis.py
# Purpose: To perform quantitative analysis of stock market using Python

# Import necessary libraries
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Function to download stock data
def download_stock_data(tickers, start_date, end_date):
    """Download stock data for given tickers within the specified date range."""
    data = yf.download(tickers, start=start_date, end=end_date)
    return data

# Function to calculate basic statistics
def calculate_statistics(data):
    """Calculate basic statistics for stock data."""
    stats = data.describe()
    return stats

# Function to calculate Simple Moving Average
def calculate_sma(data, window_size):
    """Calculate Simple Moving Average."""
    return data.rolling(window=window_size).mean()

# Function to calculate Exponential Moving Average
def calculate_ema(data, window_size):
    """Calculate Exponential Moving Average."""
    return data.ewm(span=window_size, adjust=False).mean()

# Function to calculate Relative Strength Index
def calculate_rsi(data, window_size=14):
    """Calculate Relative Strength Index."""
    delta = data.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ma_up = up.rolling(window=window_size).mean()
    ma_down = down.rolling(window=window_size).mean()

    rsi = 100 - (100 / (1 + ma_up/ma_down))
    return rsi

# Function to calculate Moving Average Convergence Divergence
def calculate_macd(data, slow=26, fast=12):
    """Calculate Moving Average Convergence Divergence."""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# Function to perform time series analysis
def time_series_analysis(data):
    """Perform time series analysis and generate plot for stock data."""
    fig = make_subplots(rows=1, cols=1)
    for column in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[column], name=column), row=1, col=1)
    fig.update_layout(title_text='Time Series of Stock Prices', xaxis_title='Date', yaxis_title='Price')
    return fig

# Function to analyze volatility
def analyze_volatility(data):
    """Analyze and plot the volatility of stock prices."""
    volatility = data.std().sort_values(ascending=False)
    fig = px.bar(volatility, x=volatility.index, y=volatility.values, labels={'y': 'Standard Deviation', 'x': 'Ticker'})
    fig.update_layout(title='Volatility of Stock Prices')
    return fig

# Function to perform correlation analysis
def correlation_analysis(data):
    """Perform correlation analysis between different stocks."""
    correlation_matrix = data.corr()
    fig = go.Figure(data=go.Heatmap(z=correlation_matrix, x=correlation_matrix.columns, y=correlation_matrix.columns))
    fig.update_layout(title='Correlation Matrix of Stock Prices')
    return fig

# Function to optimize portfolio
def optimize_portfolio(data):
    """Optimize portfolio using Efficient Frontier."""
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    return cleaned_weights

def plot_interactive_ma(data, sma, ema):
    """Plot interactive moving averages."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data, name='Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=sma, name='SMA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=ema, name='EMA', line=dict(color='green')))
    fig.update_layout(title='Stock Price with Moving Averages', xaxis_title='Date', yaxis_title='Price')
    return fig

def plot_interactive_rsi(rsi):
    """Plot interactive RSI."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rsi.index, y=rsi, name='RSI', line=dict(color='purple')))
    fig.update_layout(title='Relative Strength Index (RSI)', xaxis_title='Date', yaxis_title='RSI')
    return fig

def generate_and_save_trading_signals(stock_data, tickers):
    for ticker in tickers:
        data = stock_data[ticker]['Close']
        rsi = calculate_rsi(data, window_size=14)
        macd, signal = calculate_macd(data)

        signals = pd.DataFrame(index=data.index)
        signals['Close'] = data
        signals['RSI'] = rsi
        signals['MACD'] = macd
        signals['Signal'] = signal
        signals['Buy_Signal'] = (signals['RSI'] < 30) & (signals['MACD'] > signals['Signal'])
        signals['Sell_Signal'] = (signals['RSI'] > 70) & (signals['MACD'] < signals['Signal'])

        signals.to_csv(f'/Users/tahsinyigitgultekin/Desktop/AnalysisStock/trading_signals_{ticker}.csv')



# Main execution block
if __name__ == "__main__":
    # Define tickers and date range
    tickers = ['AAPL', 'GOOG', 'MSFT', 'NFLX']
    start_date = '2023-01-01'
    end_date = '2024-01-01'

    # Download stock data
    stock_data = download_stock_data(tickers, start_date, end_date)

    # Calculate basic statistics
    statistics = calculate_statistics(stock_data['Close'])

    # Calculate SMA, EMA, RSI, and MACD
    sma = calculate_sma(stock_data['Close'], window_size=20)
    ema = calculate_ema(stock_data['Close'], window_size=20)
    rsi = calculate_rsi(stock_data['Close'])
    macd, signal = calculate_macd(stock_data['Close'])

    # Adjust the length of all series
    length = len(stock_data['Close'])
    sma = sma.iloc[-length:]
    ema = ema.iloc[-length:]
    rsi = rsi.iloc[-length:]
    macd = macd.iloc[-length:]
    signal = signal.iloc[-length:]


    # Perform time series analysis
    time_series_fig = time_series_analysis(stock_data['Close'])

    # Analyze volatility
    volatility_fig = analyze_volatility(stock_data['Close'])

    # Perform correlation analysis
    correlation_fig = correlation_analysis(stock_data['Close'])

    # Perform portfolio optimization
    optimized_portfolio = optimize_portfolio(stock_data['Close'])
    pd.Series(optimized_portfolio).to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/optimized_portfolio.csv')


    # Plot and save interactive charts
    ma_fig = plot_interactive_ma(stock_data['Close'], sma, ema)
    rsi_fig = plot_interactive_rsi(rsi)
    ma_fig.write_image('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/ma_chart.png')
    rsi_fig.write_image('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/rsi_chart.png')

    # Save other data and analysis results to specified directory
    stock_data.to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/stock_data.csv')
    statistics.to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/statistics.csv')
    time_series_fig.write_image('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/time_series_fig.png')
    volatility_fig.write_image('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/volatility_fig.png')
    correlation_fig.write_image('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/correlation_fig.png')
    sma.to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/sma_data.csv')
    ema.to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/ema_data.csv')
    rsi.to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/rsi_data.csv')
    macd.to_csv('/Users/tahsinyigitgultekin/Desktop/AnalysisStock/macd_data.csv')
