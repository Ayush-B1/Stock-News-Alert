import yfinance as yf
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import SMAIndicator

def fetch_stock_data(symbol):
    try:
        data = yf.download(symbol, start="2020-01-01", end="2023-01-01")
        return data
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def calculate_technical_indicators(data):
    # Ensure column headers are flattened (if multi-indexed)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(-1)

    # Moving Average
    data["Moving_Avg"] = data["Close"].rolling(window=14).mean()

    # RSI Calculation
    delta = data["Close"].diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data
