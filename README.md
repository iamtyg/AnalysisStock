
# Stock Analysis Tool

## Description

This Python-based tool performs a comprehensive quantitative analysis of stock market data. Leveraging powerful libraries such as Pandas, Plotly, and yfinance, it offers functionalities ranging from downloading stock data, calculating various technical indicators, performing time series analysis, to optimizing stock portfolios using Efficient Frontier.

## Features

- **Stock Data Download**: Fetch historical stock data for specified tickers within a given date range.
- **Statistical Analysis**: Compute basic statistics for stock data.
- **Technical Indicators**: Calculate key technical indicators including SMA, EMA, RSI, and MACD.
- **Time Series Analysis**: Visualize stock price movements over time.
- **Volatility Analysis**: Analyze and visualize stock price volatility.
- **Correlation Analysis**: Examine the correlation between different stocks.
- **Portfolio Optimization**: Optimize stock portfolio using the Efficient Frontier method.
- **Interactive Charts**: Generate and save interactive charts for various analyses.

## Installation

Ensure you have Python 3.6+ installed on your system. You can then install the required dependencies by running:

```bash
pip install -r requirements.txt
```

`requirements.txt` should include:

```
yfinance
pandas
plotly
pypfopt
numpy
```

## Usage

To use this tool, simply run the `stock_analysis.py` script with Python:

```bash
python stock_analysis.py
```

Before running, ensure you configure the `tickers`, `start_date`, and `end_date` variables within the script to match your analysis requirements.

## Customization

You can customize the analysis by modifying the parameters used in the function calls within the script, such as the window size for moving averages, or the tickers list for which you're downloading data.

## Output

The script generates and saves various outputs including CSV files for statistical data, trading signals, optimized portfolio, and PNG files for interactive charts.

Ensure you set the `save_path` variable to your desired output directory.

## Contributing

Contributions to improve the tool or add new features are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the MIT License.
