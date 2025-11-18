"""
Technical indicators calculation module
"""

import pandas as pd
import numpy as np


def calculate_indicators(data, ma_short=50, ma_long=200, rsi_period=14, 
                        bollinger_period=20, bollinger_std=2):
    """
    Calculate all technical indicators
    
    Args:
        data (pd.DataFrame): Stock data with Close, High, Low, Volume columns
        ma_short (int): Short-term moving average period
        ma_long (int): Long-term moving average period
        rsi_period (int): RSI calculation period
        bollinger_period (int): Bollinger Bands period
        bollinger_std (float): Bollinger Bands standard deviation multiplier
        
    Returns:
        pd.DataFrame: Data with all indicators added
    """
    print("Calculating technical indicators...")
    
    # Moving Averages
    data[f'MA{ma_short}'] = data['Close'].rolling(window=ma_short).mean()
    data[f'MA{ma_long}'] = data['Close'].rolling(window=ma_long).mean()
    
    # RSI
    delta = data['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    data['BB_middle'] = data['Close'].rolling(window=bollinger_period).mean()
    data['BB_std'] = data['Close'].rolling(window=bollinger_period).std()
    data['BB_upper'] = data['BB_middle'] + (bollinger_std * data['BB_std'])
    data['BB_lower'] = data['BB_middle'] - (bollinger_std * data['BB_std'])
    
    # MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_hist'] = data['MACD'] - data['MACD_signal']
    
    # Volume
    data['Volume_MA20'] = data['Volume'].rolling(window=20).mean()
    
    # Daily Returns
    data['Daily_Return'] = data['Close'].pct_change() * 100
    
    # EMAs for trend detection
    data['EMA10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA30'] = data['Close'].ewm(span=30, adjust=False).mean()
    
    print("Indicators calculated")
    return data
